import os
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import trange

from gcbfplus.env import make_env
from gcbfplus.algo.dec_share_cbf import DecShareCBF
from gcbfplus.utils.graph import GraphsTuple


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Args:
    env: str = "DoubleIntegrator"
    algo: str = "dec_share_cbf"

    num_agents: int = 8
    obs: int = 0
    area_size: float = 4.0
    max_step: int = 400
    max_travel: Optional[float] = None

    path: str = (
        "/home/sharma/Projects/gcbfplus/pretrained_diffuser/"
        "DoubleIntegrator/gcbfdiffuser"
    )
    step: Optional[int] = None

    seed: int = 1234
    debug: bool = False
    cpu: bool = False
    radius_ratio: float = 1.0

    # Circular position-exchange t


    save_animation: bool = True
    animation_file: str = "position_swap_graph.gif"

    # Per-agent minimum-distance plot
    save_distance_plot: bool = True
    distance_plot_file: str = "per_agent_min_distance.png"

    results_file: str = "position_swap_graph.npz"

    def __post_init__(self):
        assert self.num_agents % 2 == 0, (
            "Position swapping requires an even number of agents."
        )


# ============================================================
# GRAPH / NODE HELPERS
# ============================================================

def get_agent_ids(graph: GraphsTuple, num_agents: int) -> jnp.ndarray:
    """Agents use node_type == 0."""
    return jnp.where(
        graph.node_type == 0,
        size=num_agents,
        fill_value=0,
    )[0]


def get_goal_ids(graph: GraphsTuple, num_agents: int) -> jnp.ndarray:
    """Goals are assumed to use node_type == 1."""
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
    Agent i starts on a circle and targets the location initially
    occupied by agent i + N/2.
    """
    center = jnp.array([
        area_size / 2.0,
        area_size / 2.0,
    ])

    radius = radius_ratio * area_size

    theta = (
        2.0 * jnp.pi
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
    Replaces random agent positions/goals with deterministic
    position-exchange initialization.

    Agent state:
        [x, y, vx, vy]

    Goal state:
        [gx, gy, 0, 0]
    """
    n_agents = env.num_agents

    agent_ids = get_agent_ids(graph, n_agents)
    goal_ids = get_goal_ids(graph, n_agents)

    if np.any(np.asarray(goal_ids) == 0) and n_agents > 1:
        raise RuntimeError(
            "Goal nodes were not found as node_type == 1. "
            "Print graph.node_type and update get_goal_ids()."
        )

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

    graph_new = env.add_edge_feats(graph, new_states)

    return graph_new, start_xy, goal_xy


def snapshot_graph(graph: GraphsTuple):
    """Convert graph values to NumPy for animation."""
    return {
        "states": np.asarray(graph.states).copy(),
        "node_type": np.asarray(graph.node_type).copy(),
        "senders": np.asarray(graph.senders).copy(),
        "receivers": np.asarray(graph.receivers).copy(),
    }


# ============================================================
# EDGE HELPERS FOR ANIMATION
# ============================================================

def active_agent_edges(
    agent_xy: np.ndarray,
    comm_radius: float,
):
    """
    Exact DoubleIntegrator rule:

        edge(i, j) exists when ||p_i - p_j|| < comm_radius

    Returns undirected visual edges.
    """
    n_agents = agent_xy.shape[0]
    edges = []

    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            distance = np.linalg.norm(
                agent_xy[i] - agent_xy[j]
            )

            if distance < comm_radius:
                edges.append((i, j))

    return edges


def active_edges_from_snapshot(graph_snapshot):
    """
    Remove dummy-node edges.

    In GraphsTuple.to_padded(), inactive edges are redirected
    to a padding node with node_type == -1.
    """
    node_type = graph_snapshot["node_type"]
    senders = graph_snapshot["senders"]
    receivers = graph_snapshot["receivers"]

    valid_mask = (
        (node_type[senders] != -1)
        & (node_type[receivers] != -1)
    )

    return senders[valid_mask], receivers[valid_mask]


def unique_undirected_edges(senders, receivers):
    """Merge directed pairs i->j and j->i into one visual edge."""
    edges = set()

    for sender, receiver in zip(senders, receivers):
        sender = int(sender)
        receiver = int(receiver)

        if sender != receiver:
            edges.add(tuple(sorted((sender, receiver))))

    return list(edges)


def split_graph_goal_edges(graph_snapshot):
    """
    Extract only active agent-goal graph edges.

    node_type:
        0 = agent
        1 = goal
       -1 = padding
    """
    node_type = graph_snapshot["node_type"]

    senders, receivers = active_edges_from_snapshot(
        graph_snapshot
    )

    goal_edges = []

    for sender, receiver in unique_undirected_edges(
        senders,
        receivers,
    ):
        sender_type = node_type[sender]
        receiver_type = node_type[receiver]

        is_agent_goal = (
            (sender_type == 0 and receiver_type == 1)
            or (sender_type == 1 and receiver_type == 0)
        )

        if is_agent_goal:
            goal_edges.append((sender, receiver))

    return goal_edges


# ============================================================
# METRICS
# ============================================================

def pairwise_min_distance(agent_states: np.ndarray) -> float:
    """Smallest planar centre-to-centre separation."""
    xy = agent_states[:, :2]
    n_agents = xy.shape[0]

    minimum_distance = np.inf

    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            distance = np.linalg.norm(xy[i] - xy[j])
            minimum_distance = min(minimum_distance, distance)

    return float(minimum_distance)


def per_agent_min_distance(agent_states: np.ndarray) -> np.ndarray:
    """
    For each agent i, return its minimum planar center-to-center
    distance to any other agent j != i.

    Output shape:
        (num_agents,)
    """
    xy = agent_states[:, :2]
    n_agents = xy.shape[0]

    min_distances = np.full(n_agents, np.inf)

    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                continue

            distance = np.linalg.norm(xy[i] - xy[j])
            min_distances[i] = min(min_distances[i], distance)

    return min_distances


def plot_per_agent_min_distance(
    results,
    dt: float,
    car_radius: float,
    filename: str,
):
    """
    Plot one curve per agent:

        d_i(t) = min_{j != i} ||p_i(t) - p_j(t)||.

    The dashed red line is the physical collision boundary:

        d_collision = 2 * car_radius.
    """
    distances = results["per_agent_min_distance"]
    n_steps, n_agents = distances.shape
    time = np.arange(n_steps) * dt

    fig, ax = plt.subplots(figsize=(8, 4.5))

    for i in range(n_agents):
        ax.plot(
            time,
            distances[:, i],
            linewidth=2.0,
            label=str(i),
        )

    ax.axhline(
        2.0 * car_radius,
        color="red",
        linestyle="--",
        linewidth=2.0,
        label="Collision boundary",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Min distance (m)")
    ax.set_title("Minimum Distance to Other Agents")
    ax.grid(True, alpha=0.30)

    ax.legend(
        title="Agent",
        ncol=2,
        fontsize=9,
        loc="upper right",
    )

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved minimum-distance plot: {filename}")


# ============================================================
# PERFECT-STATE SIMULATION ROLLOUT
# ============================================================

def rollout_position_swap_decentralized_qp(
    env,
    controller: DecShareCBF,
    key,
    rollout_length: int,
    radius_ratio: float,
    relax_penalty: float = 1e3,
    use_jit: bool = True,
):
    """
    Perfect-state rollout using the decentralized shared CBF-QP controller.

Each agent solves its own local QP using its k nearest CBF constraints.

        graph_t
          -> decentralized shared CBF-QP
          -> action_t, relaxation_t
          -> env.step(graph_t, action_t)
          -> graph_{t+1}

    No learned policy, checkpoint, Vicon delay, or added noise is used.
    """
    graph_true = env.reset(key)

    graph_true, start_xy, goal_xy = set_position_swap_graph(
        env=env,
        graph=graph_true,
        radius_ratio=radius_ratio,
    )

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
    relaxation_log = []
    rewards_log = []
    costs_log = []
    done_log = []
    unsafe_log = []
    finish_log = []

    min_distance_log = [
        pairwise_min_distance(initial_agent_states)
    ]

    per_agent_min_distance_log = [
        per_agent_min_distance(initial_agent_states)
    ]

    if use_jit:
        qp_fn = jax.jit(
            lambda graph: controller.get_qp_action(
                graph,
                relax_penalty=relax_penalty,
            )
        )
    else:
        qp_fn = lambda graph: controller.get_qp_action(
            graph,
            relax_penalty=relax_penalty,
        )

    for step_idx in trange(
        rollout_length,
        ncols=80,
        desc="Decentralized shared CBF-QP",
    ):
        action, relaxation = qp_fn(graph_true)

        graph_next, reward, cost, done, info = env.step(
            graph_true,
            action,
            get_eval_info=True,
        )

        unsafe = env.collision_mask(graph_next)
        finish = env.finish_mask(graph_next)

        next_agent_states = np.asarray(
            jax.device_get(graph_next.states[agent_ids])
        )

        true_states_log.append(next_agent_states)
        graph_log.append(snapshot_graph(graph_next))

        actions_log.append(
            np.asarray(jax.device_get(action))
        )
        relaxation_log.append(
            np.asarray(jax.device_get(relaxation))
        )
        rewards_log.append(
            np.asarray(jax.device_get(reward))
        )
        costs_log.append(
            np.asarray(jax.device_get(cost))
        )
        done_log.append(
            np.asarray(jax.device_get(done))
        )
        unsafe_log.append(
            np.asarray(jax.device_get(unsafe))
        )
        finish_log.append(
            np.asarray(jax.device_get(finish))
        )

        min_distance_log.append(
            pairwise_min_distance(next_agent_states)
        )

        per_agent_min_distance_log.append(
            per_agent_min_distance(next_agent_states)
        )

        if step_idx % 20 == 0:
            action_np = np.asarray(jax.device_get(action))
            relaxation_np = np.asarray(
                jax.device_get(relaxation)
            )
            finish_np = np.asarray(
                jax.device_get(finish),
                dtype=bool,
            )
            unsafe_np = np.asarray(
                jax.device_get(unsafe),
                dtype=bool,
            )

            print(
                f"step={step_idx:04d}, "
                f"max component |u|="
                f"{np.max(np.abs(action_np)):.4f}, "
                f"max ||u||="
                f"{np.max(np.linalg.norm(action_np, axis=-1)):.4f}, "
                f"mean relaxation="
                f"{np.mean(relaxation_np):.6f}, "
                f"max relaxation="
                f"{np.max(relaxation_np):.6f}, "
                f"min distance="
                f"{min_distance_log[-1]:.4f}, "
                f"unsafe="
                f"{bool(np.any(unsafe_np))}, "
                f"finished="
                f"{int(np.sum(finish_np))}/{env.num_agents}"
            )

        graph_true = graph_next

        if bool(np.any(np.asarray(jax.device_get(done)))):
            break

    if not actions_log:
        raise RuntimeError(
            "The rollout ended before any action was generated."
        )

    return {
        "start_xy": np.asarray(start_xy),
        "goal_xy": np.asarray(goal_xy),
        "true_states": np.stack(true_states_log, axis=0),
        "actions": np.stack(actions_log, axis=0),
        "relaxation": np.stack(relaxation_log, axis=0),
        "rewards": np.stack(rewards_log, axis=0),
        "costs": np.stack(costs_log, axis=0),
        "done": np.stack(done_log, axis=0),
        "unsafe": np.stack(unsafe_log, axis=0),
        "finish": np.stack(finish_log, axis=0),
        "min_distance": np.asarray(min_distance_log),
        "per_agent_min_distance": np.stack(
            per_agent_min_distance_log,
            axis=0,
        ),
        "graph_log": graph_log,
    }


# ============================================================
# ANIMATION
# ============================================================

def animate_position_swap_graph(
    results,
    area_size: float,
    filename: str,
    comm_radius: float,
    interval_ms: int = 50,
):
    """
    Blue circles: agents
    Green circles: goals
    Green lines: permanent agent-goal graph edges
    Black lines: dynamic communication edges
    """
    graph_log = results["graph_log"]
    true_xy = results["true_states"][:, :, :2]

    n_frames = len(graph_log)
    n_agents = true_xy.shape[1]

    first_graph = graph_log[0]
    first_states = first_graph["states"]
    first_node_type = first_graph["node_type"]

    goal_ids = np.where(first_node_type == 1)[0]

    if len(goal_ids) != n_agents:
        raise RuntimeError(
            f"Expected {n_agents} goal nodes, found {len(goal_ids)}. "
            "Check goal node type."
        )

    goal_xy = first_states[goal_ids, :2]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(0.0, area_size)
    ax.set_ylim(0.0, area_size)
    ax.set_aspect("equal")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        "Decentralized shared CBF-QP Position Exchange with Dynamic Graph Edges"
    )
    ax.grid(True, alpha=0.25)

    ax.scatter(
        goal_xy[:, 0],
        goal_xy[:, 1],
        s=180,
        c="limegreen",
        edgecolors="forestgreen",
        linewidths=1.5,
        zorder=2,
        label="Goals",
    )

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

    goal_line_artists = []
    proximity_line_artists = []

    status_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=11,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )

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
        label="Communication edges",
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

        agent_scatter.set_offsets(agent_xy)

        for i, label in enumerate(agent_labels):
            label.set_position(
                (agent_xy[i, 0], agent_xy[i, 1])
            )

        for i, line in enumerate(trajectory_lines):
            line.set_data(
                true_xy[:frame + 1, i, 0],
                true_xy[:frame + 1, i, 1],
            )

        remove_current_edges()

        # Permanent green agent-goal edges from graph topology.
        green_edges = split_graph_goal_edges(
            current_graph
        )

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

        # Black communication edges use exactly:
        # distance < env._params["comm_radius"]
        black_edges = active_agent_edges(
            agent_xy=agent_xy,
            comm_radius=comm_radius,
        )

        for i, j in black_edges:
            p_i = agent_xy[i]
            p_j = agent_xy[j]

            line, = ax.plot(
                [p_i[0], p_j[0]],
                [p_i[1], p_j[1]],
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
            f"Step: {frame:04d}\n"
            f"Min distance: "
            f"{results['min_distance'][frame]:.3f} m\n"
            f"Comm radius: {comm_radius:.3f} m"
        )

        return [
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

    animation.save(
        filename,
        writer=PillowWriter(
            fps=max(1, int(1000 / interval_ms))
        ),
    )

    plt.close(fig)

    print(f"Saved animation: {filename}")


# ============================================================
# MAIN TEST
# ============================================================

def test(args: Args):
    print(
        f"> Running decentralized shared CBF-QP position-swap test: {args}"
    )

    # These environment variables are most effective when exported
    # in the shell before launching Python.
    os.environ.setdefault(
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "false",
    )

    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"

    if args.debug:
        jax.config.update("jax_disable_jit", True)

    np.random.seed(args.seed)

    env = make_env(
        env_id=args.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        area_size=args.area_size,
        max_step=args.max_step,
        max_travel=args.max_travel,
    )

    controller = DecShareCBF(
        env=env,
        node_dim=env.node_dim,
        edge_dim=env.edge_dim,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        n_agents=env.num_agents,
        alpha=1.0,
    )

    debug_graph = env.reset(
        jr.PRNGKey(args.seed)
    )

    action_lb, action_ub = env.action_lim()

    print("Graph diagnostics:")
    print("states shape:", debug_graph.states.shape)
    print(
        "node types:",
        np.unique(np.asarray(debug_graph.node_type)),
    )
    print(
        "comm_radius:",
        float(env._params["comm_radius"]),
    )
    print(
        "car_radius:",
        float(env._params["car_radius"]),
    )
    print(
        "action lower bound:",
        np.asarray(action_lb),
    )
    print(
        "action upper bound:",
        np.asarray(action_ub),
    )

    results = rollout_position_swap_decentralized_qp(
        env=env,
        controller=controller,
        key=jr.PRNGKey(args.seed),
        rollout_length=args.max_step,
        radius_ratio=args.radius_ratio,
        relax_penalty=1e3,
        use_jit=not args.debug,
    )

    actions = np.asarray(results["actions"])
    relaxation = np.asarray(results["relaxation"])
    unsafe = np.asarray(results["unsafe"], dtype=bool)
    finish = np.asarray(results["finish"], dtype=bool)

    action_component_abs = np.abs(actions)
    action_norms = np.linalg.norm(actions, axis=-1)
    finish_counts = np.sum(finish, axis=-1)

    print("========== DECENTRALIZED SHARED CBF-QP RESULTS ==========")
    print(
        "True-state trajectory:",
        results["true_states"].shape,
    )
    print(
        "Action trajectory:",
        actions.shape,
    )
    print(
        "Relaxation trajectory:",
        relaxation.shape,
        "(steps, agents, k)",
    )
    print(
        "Unsafe rollout:",
        bool(np.any(unsafe)),
    )
    print(
        "Any agent reached a goal:",
        bool(np.any(finish)),
    )
    print(
        "All agents simultaneously reached goals:",
        bool(np.any(finish_counts == env.num_agents)),
    )
    print(
        "Maximum agents finished simultaneously:",
        int(np.max(finish_counts)),
        "/",
        env.num_agents,
    )
    print(
        "Agents inside goal region at final step:",
        int(finish_counts[-1]),
        "/",
        env.num_agents,
    )
    print(
        "Maximum action component magnitude:",
        float(np.max(action_component_abs)),
    )
    print(
        "Maximum action vector norm:",
        float(np.max(action_norms)),
    )
    print(
        "Maximum relaxation:",
        float(np.max(relaxation)),
    )
    print(
        "Mean relaxation:",
        float(np.mean(relaxation)),
    )
    print(
        "Maximum relaxation per agent:",
        np.max(relaxation, axis=(0, 2)),
    )
    print(
        "Mean relaxation per agent:",
        np.mean(relaxation, axis=(0, 2)),
    )
    print(
        "Maximum cost:",
        float(np.max(results["costs"])),
    )
    print(
        "Minimum pairwise distance:",
        float(np.min(results["min_distance"])),
    )
    print(
        "Collision boundary:",
        2.0 * float(env._params["car_radius"]),
    )

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
            comm_radius=float(
                env._params["comm_radius"]
            ),
            interval_ms=50,
        )

    if args.save_distance_plot:
        plot_per_agent_min_distance(
            results=results,
            dt=float(env.dt),
            car_radius=float(
                env._params["car_radius"]
            ),
            filename=args.distance_plot_file,
        )

    return results


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    args = Args(
        env="DoubleIntegrator",
        algo="dec_share_cbf",
        num_agents=8,
        obs=0,
        area_size=4.0,
        max_step=1000,
        debug=False,
        cpu=False,
        save_animation=True,
        animation_file=(
            "dec_share_cbf_qp_position_swap.gif"
        ),
        save_distance_plot=True,
        distance_plot_file=(
            "dec_share_cbf_qp_per_agent_min_distance.png"
        ),
        results_file=(
            "dec_share_cbf_qp_position_swap.npz"
        ),
        radius_ratio=0.25,  # 0.25 * 4.0 m = 1.0 m radius
    )

    results = test(args)
