import os
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import trange


from gcbfplus.algo import make_algo
from gcbfplus.env import make_env
from gcbfplus.utils.graph import GraphsTuple


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Args:
    # Environment
    env: str = "DoubleIntegrator" # Dynamics
    algo: str = "gcbf_diffuser" # Algorithm
    num_agents: int = 8 # Number of agents
    obs: int = 0 # No obstacle
    area_size: float = 4.0 # Area of the drone workspace in the lab
    max_step: int = 400 # Time Horizon
    max_travel: Optional[float] = None

    # Pretrained model
    path: str = (
        "/home/sharma/Projects/gcbfplus/pretrained_diffuser/" #change
        "DoubleIntegrator/gcbfdiffuser"
    ) # location of pretrained models
    step: Optional[int] = None

    # Runtime
    seed: int = 1234
    debug: bool = False
    cpu: bool = False

    # Position exchange
    radius_ratio: float = 0.35

    # Outputs
    save_animation: bool = True
    animation_file: str = "position_swap_graph.gif"
    results_file: str = "position_swap_graph.npz"

    def __post_init__(self):
        assert self.num_agents % 2 == 0, (
            "Position swapping requires an even number of agents."
        )


# ============================================================
# GRAPH HELPERS
# ============================================================

def get_agent_ids(graph: GraphsTuple, num_agents: int) -> jnp.ndarray:
    """
    Assumption:
        node_type == 0 corresponds to drone / agent nodes.
    """
    return jnp.where(
        graph.node_type == 0,
        size=num_agents,
        fill_value=0,
    )[0]


def get_goal_ids(graph: GraphsTuple, num_agents: int) -> jnp.ndarray:
    """
    Assumption:
        node_type == 1 corresponds to goal nodes.
    """
    return jnp.where(
        graph.node_type == 1,
        size=num_agents,
        fill_value=0,
    )[0]


def make_position_swap_configuration(
    num_agents: int,
    area_size: float,
    radius_ratio: float,
):
    """
    Place N drones on a circle.

    Drone i receives the target occupied initially by drone i + N/2.
    """
    center = jnp.array([
        area_size / 2.0,
        area_size / 2.0,
    ])

    radius = radius_ratio * area_size

    theta = (
        2.0
        * jnp.pi
        * jnp.arange(num_agents)
        / num_agents
    )

    start_xy = center + radius * jnp.stack(
        [
            jnp.cos(theta),
            jnp.sin(theta),
        ],
        axis=1,
    )

    goal_xy = jnp.roll(
        start_xy,
        shift=num_agents // 2,
        axis=0,
    )

    return start_xy, goal_xy


def set_position_swap_graph(
    env,
    graph: GraphsTuple,
    radius_ratio: float,
):
    """
    Overwrite the reset graph with a deterministic circular
    position-exchange configuration.

    Agent states:
        [x, y, vx, vy]

    Goal states:
        [goal_x, goal_y, 0, 0]
    """
    n_agents = env.num_agents

    agent_ids = get_agent_ids(graph, n_agents)
    goal_ids = get_goal_ids(graph, n_agents)

    start_xy, goal_xy = make_position_swap_configuration(
        num_agents=n_agents,
        area_size=env.area_size,
        radius_ratio=radius_ratio,
    )

    agent_states = jnp.concatenate(
        [
            start_xy,
            jnp.zeros((n_agents, 2)),
        ],
        axis=1,
    )

    goal_states = jnp.concatenate(
        [
            goal_xy,
            jnp.zeros((n_agents, 2)),
        ],
        axis=1,
    )

    new_states = graph.states.at[agent_ids].set(agent_states)
    new_states = new_states.at[goal_ids].set(goal_states)

    # Recalculate edge features after changing agent and goal states.
    graph_new = env.add_edge_feats(graph, new_states)

    return graph_new, start_xy, goal_xy


def snapshot_graph(graph: GraphsTuple):
    """
    Save the graph data needed for visualisation.

    We save it as NumPy because the animation runs outside JAX.
    """
    return {
        "states": np.asarray(graph.states).copy(),
        "node_type": np.asarray(graph.node_type).copy(),
        "senders": np.asarray(graph.senders).copy(),
        "receivers": np.asarray(graph.receivers).copy(),
    }


def unique_undirected_edges(senders, receivers):
    """
    Convert directed edge pairs into unique undirected pairs.

    Example:
        1 -> 4 and 4 -> 1
    become:
        (1, 4)
    """
    edges = set()

    for sender, receiver in zip(senders, receivers):
        sender = int(sender)
        receiver = int(receiver)

        if sender == receiver:
            continue

        edges.add(tuple(sorted((sender, receiver))))

    return list(edges)


def split_graph_edges(graph_snapshot):
    """
    Separate graph edges into:

    green_edges:
        agent <-> goal links

    black_edges:
        agent <-> agent proximity / communication links
    """
    node_type = graph_snapshot["node_type"]
    senders = graph_snapshot["senders"]
    receivers = graph_snapshot["receivers"]

    green_edges = []
    black_edges = []

    for sender, receiver in unique_undirected_edges(senders, receivers):
        sender_type = node_type[sender]
        receiver_type = node_type[receiver]

        # Agent-goal edge.
        if (
            (sender_type == 0 and receiver_type == 1)
            or (sender_type == 1 and receiver_type == 0)
        ):
            green_edges.append((sender, receiver))

        # Agent-agent proximity edge.
        elif sender_type == 0 and receiver_type == 0:
            black_edges.append((sender, receiver))

    return green_edges, black_edges


# ============================================================
# METRICS
# ============================================================

def pairwise_min_distance(agent_states: np.ndarray) -> float:
    """
    Return smallest pairwise planar distance between all drones.
    """
    xy = agent_states[:, :2]
    n_agents = xy.shape[0]

    minimum_distance = np.inf

    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            distance = np.linalg.norm(xy[i] - xy[j])
            minimum_distance = min(minimum_distance, distance)

    return float(minimum_distance)


# ============================================================
# ORIGINAL STATE TRAJECTORY ROLLOUT
# ============================================================

def rollout_position_swap_original_dynamics(
    env,
    act_fn,
    key,
    rollout_length: int,
    radius_ratio: float,
):
    """
    Original GCBF+ simulation:

        true graph
            -> policy
            -> action
            -> env.step(...)
            -> next true graph

    No noisy estimates.
    No delay.
    No hardware model.
    """

    # Environment creates initial graph.
    graph_true = env.reset(key)

    # Replace random initialization with circular position exchange.
    graph_true, start_xy, goal_xy = set_position_swap_graph(
        env=env,
        graph=graph_true,
        radius_ratio=radius_ratio,
    )

    # Extract agent IDs
    agent_ids = get_agent_ids(
        graph_true,
        env.num_agents,
    )

    initial_agent_states = np.asarray(
        graph_true.states[agent_ids]
    )

    true_states_log = [initial_agent_states]
    graph_log = [snapshot_graph(graph_true)]

    actions_log = []
    rewards_log = []
    costs_log = []
    done_log = []
    unsafe_log = []
    finish_log = []

    min_distance_log = [
        pairwise_min_distance(initial_agent_states)
    ]

    for _ in trange(rollout_length, ncols=80):
        # The policy receives the exact original graph.
        action = act_fn(graph_true)

        # Original DoubleIntegrator simulation.
        graph_next, reward, cost, done, info = env.step(
            graph_true,
            action,
            get_eval_info=True,
        )

        unsafe = env.collision_mask(graph_next)
        finish = env.finish_mask(graph_next)

        next_agent_states = np.asarray(
            graph_next.states[agent_ids]
        )

        true_states_log.append(next_agent_states)
        graph_log.append(snapshot_graph(graph_next))

        actions_log.append(np.asarray(action))
        rewards_log.append(np.asarray(reward))
        costs_log.append(np.asarray(cost))
        done_log.append(np.asarray(done))
        unsafe_log.append(np.asarray(unsafe))
        finish_log.append(np.asarray(finish))

        min_distance_log.append(
            pairwise_min_distance(next_agent_states)
        )

        graph_true = graph_next

        if bool(np.any(np.asarray(done))):
            break

    true_states = np.stack(true_states_log, axis=0)

    return {
        "start_xy": np.asarray(start_xy),
        "goal_xy": np.asarray(goal_xy),
        "true_states": true_states,
        "actions": np.stack(actions_log, axis=0),
        "rewards": np.stack(rewards_log, axis=0),
        "costs": np.stack(costs_log, axis=0),
        "done": np.stack(done_log, axis=0),
        "unsafe": np.stack(unsafe_log, axis=0),
        "finish": np.stack(finish_log, axis=0),
        "min_distance": np.asarray(min_distance_log),
        "graph_log": graph_log,
    }


# ============================================================
# GRAPH ANIMATION (function to generate positions wapping animation)
# ============================================================

def animate_position_swap_graph(
    results,
    area_size: float,
    filename: str,
    interval_ms: int = 50,
):
    """
    Animation style:

    Blue nodes:
        drones / agents

    Green nodes:
        assigned goals

    Green edges:
        agent-goal task edges

    Black edges:
        active agent-agent proximity graph edges
    """
    graph_log = results["graph_log"]
    true_xy = results["true_states"][:, :, :2]

    n_frames = len(graph_log)
    n_agents = true_xy.shape[1]

    first_graph = graph_log[0]

    first_states = first_graph["states"]
    first_node_type = first_graph["node_type"]

    goal_ids = np.where(first_node_type == 1)[0]
    goal_xy = first_states[goal_ids, :2]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(0.0, area_size)
    ax.set_ylim(0.0, area_size)
    ax.set_aspect("equal")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    ax.set_title(
        "GCBF-Diffuser Position Exchange with Dynamic Graph Edges"
    )

    ax.grid(True, alpha=0.25)

    # Green goal nodes.
    goal_scatter = ax.scatter(
        goal_xy[:, 0],
        goal_xy[:, 1],
        s=180,
        c="limegreen",
        edgecolors="forestgreen",
        linewidths=1.5,
        zorder=2,
        label="Goals",
    )

    # Blue agent nodes.
    agent_scatter = ax.scatter(
        true_xy[0, :, 0],
        true_xy[0, :, 1],
        s=180,
        c="dodgerblue",
        edgecolors="navy",
        linewidths=1.5,
        zorder=5,
        label="Agents",
    )

    # Drone ID labels.
    agent_labels = []

    for i in range(n_agents):
        label = ax.text(
            true_xy[0, i, 0],
            true_xy[0, i, 1],
            str(i),
            color="black",
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
        )
        agent_labels.append(label)

    # Trajectory lines.
    trajectory_lines = []

    for i in range(n_agents):
        line, = ax.plot(
            true_xy[:1, i, 0],
            true_xy[:1, i, 1],
            color="dodgerblue",
            alpha=0.35,
            linewidth=1.0,
            zorder=1,
        )

        trajectory_lines.append(line)

    # Edge lists get redrawn each frame.
    goal_line_artists = []
    proximity_line_artists = []

    status_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=12,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )

    # Legend placeholders.
    ax.plot(
        [],
        [],
        color="limegreen",
        linewidth=2.0,
        label="Agent-goal edges",
    )

    ax.plot(
        [],
        [],
        color="black",
        linewidth=1.5,
        label="Proximity edges",
    )

    ax.legend(loc="lower right")

    def remove_current_edges():
        nonlocal goal_line_artists
        nonlocal proximity_line_artists

        for line in goal_line_artists:
            line.remove()

        for line in proximity_line_artists:
            line.remove()

        goal_line_artists = []
        proximity_line_artists = []

    def update(frame):
        nonlocal goal_line_artists
        nonlocal proximity_line_artists

        current_graph = graph_log[frame]

        states = current_graph["states"]
        node_type = current_graph["node_type"]

        agent_ids = np.where(node_type == 0)[0]
        agent_xy = states[agent_ids, :2]

        # Move blue agent markers.
        agent_scatter.set_offsets(agent_xy)

        # Move labels.
        for i, label in enumerate(agent_labels):
            label.set_position(
                (
                    agent_xy[i, 0],
                    agent_xy[i, 1],
                )
            )

        # Extend trajectories.
        for i, line in enumerate(trajectory_lines):
            line.set_data(
                true_xy[:frame + 1, i, 0],
                true_xy[:frame + 1, i, 1],
            )

        # Remove prior edge lines.
        remove_current_edges()

        green_edges, black_edges = split_graph_edges(
            current_graph
        )

        # Permanent task / goal edges.
        for sender, receiver in green_edges:
            p_sender = states[sender, :2]
            p_receiver = states[receiver, :2]

            line, = ax.plot(
                [p_sender[0], p_receiver[0]],
                [p_sender[1], p_receiver[1]],
                color="limegreen",
                alpha=0.50,
                linewidth=2.0,
                zorder=2,
            )

            goal_line_artists.append(line)

        # Dynamic close-proximity / communication edges.
        for sender, receiver in black_edges:
            p_sender = states[sender, :2]
            p_receiver = states[receiver, :2]

            line, = ax.plot(
                [p_sender[0], p_receiver[0]],
                [p_sender[1], p_receiver[1]],
                color="black",
                alpha=0.70,
                linewidth=1.5,
                zorder=4,
            )

            proximity_line_artists.append(line)

        if frame == 0:
            reward = 0.0
            cost = 0.0
            unsafe = False
        else:
            reward = results["rewards"][frame - 1]
            cost = results["costs"][frame - 1]
            unsafe = results["unsafe"][frame - 1]

        status_text.set_text(
            f"Cost: {float(np.asarray(cost)):.4f}, "
            f"Reward: {float(np.asarray(reward)):.4f}\n"
            f"Unsafe: {bool(np.any(np.asarray(unsafe)))}\n"
            f"kk={frame:04d}\n"
            f"Min dist: {results['min_distance'][frame]:.3f}"
        )

        return [
            goal_scatter,
            agent_scatter,
            status_text,
            *agent_labels,
            *trajectory_lines,
            *goal_line_artists,
            *proximity_line_artists,
        ]

    animation = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=interval_ms,
        repeat=False,
        blit=False,
    )

    fps = max(1, int(1000 / interval_ms))

    animation.save(
        filename,
        writer=PillowWriter(fps=fps),
    )

    plt.close(fig)

    print(f"Saved graph animation: {filename}")


# ============================================================
# MAIN TEST
# ============================================================

def test(args: Args):
    print(f"> Running position-swap test: {args}")

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"

    if args.debug:
        jax.config.update("jax_disable_jit", True)

    np.random.seed(args.seed)

    config_path = os.path.join(
        args.path,
        "config.yaml",
    )

    with open(config_path, "r") as f:
        config = yaml.load(
            f,
            Loader=yaml.UnsafeLoader,
        )

    env = make_env(
        env_id=config.env if args.env is None else args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        area_size=args.area_size,
        max_step=args.max_step,
        max_travel=args.max_travel,
    )

    model_path = os.path.join(
        args.path,
        "models",
    )

    if args.step is None:
        checkpoints = [
            int(name)
            for name in os.listdir(model_path)
            if name.isdigit()
        ]

        if len(checkpoints) == 0:
            raise FileNotFoundError(
                f"No numeric checkpoints found in: {model_path}"
            )

        step = max(checkpoints)
    else:
        step = args.step

    print("Loading model checkpoint:", step)

    algo = make_algo(
        algo=config.algo,
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        gnn_layers=config.gnn_layers,
        batch_size=config.batch_size,
        buffer_size=config.buffer_size,
        horizon=config.horizon,
        lr_actor=config.lr_actor,
        lr_cbf=config.lr_cbf,
        alpha=config.alpha,
        eps=0.02,
        inner_epoch=8,
        loss_action_coef=config.loss_action_coef,
        loss_unsafe_coef=config.loss_unsafe_coef,
        loss_safe_coef=config.loss_safe_coef,
        loss_h_dot_coef=config.loss_h_dot_coef,
        max_grad_norm=2.0,
        seed=config.seed,
    )

    algo.load(model_path, step)

    # Exact original policy.
    act_fn = jax.jit(algo.act)

    debug_graph = env.reset(
        jr.PRNGKey(args.seed)
    )

    print("\nGraph diagnostics:")
    print("states shape:", debug_graph.states.shape)
    # print("node_type:", np.asarray(debug_graph.node_type))
    print(
        "unique node types:",
        np.unique(np.asarray(debug_graph.node_type)),
    )
    print(
        "number of graph edges:",
        len(np.asarray(debug_graph.senders)),
    )

    results = rollout_position_swap_original_dynamics(
        env=env,
        act_fn=act_fn,
        key=jr.PRNGKey(args.seed),
        rollout_length=args.max_step,
        radius_ratio=args.radius_ratio,
    )

    print("\n========== RESULTS ==========")
    print("True-state trajectory:", results["true_states"].shape)
    print("Action trajectory:", results["actions"].shape)
    print("Unsafe rollout:", bool(np.any(results["unsafe"])))
    print("Goal reached:", bool(np.any(results["finish"])))
    print("Maximum cost:", float(np.max(results["costs"])))
    print(
        "Minimum pairwise distance:",
        float(np.min(results["min_distance"])),
    )

    # graph_log is not saved because each time step can have a
    # different number of proximity edges.
    results_to_save = {
        key: value
        for key, value in results.items()
        if key != "graph_log"
    }

    np.savez(
        args.results_file,
        **results_to_save,
    )

    print(f"Saved rollout data: {args.results_file}")

    if args.save_animation:
        animate_position_swap_graph(
            results=results,
            area_size=args.area_size,
            filename=args.animation_file,
            interval_ms=50,
        )

    return results


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    args = Args(
        env="DoubleIntegrator",
        algo="gcbf_diffuser",
        num_agents=8,
        obs=0,
        area_size=4.0,
        max_step=400,
        debug=False,
        path=(
            "/home/sharma/Projects/gcbfplus/pretrained_diffuser/"
            "DoubleIntegrator/gcbfdiffuser"
        ),
        radius_ratio=0.35,
        save_animation=True,
        animation_file="position_swap_graph.gif",
        results_file="position_swap_graph.npz",
    )

    results = test(args)