import os
import csv
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
    env: str = "DoubleIntegrator"
    algo: str = "gcbf_diffuser"

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


    # Multi-case evaluation
    n_trials: int = 25
    output_dir: str = "position_swap_batch"

    # Randomize the global rotation of each circular swap case.
    randomize_angle: bool = True

    # Optional radius jitter. A value of 0.03 means that each trial uses
    # radius_ratio +/- up to 0.03, while remaining inside the workspace.
    radius_ratio_jitter: float = 0.0

    # Save one GIF and one minimum-distance plot for every trial.
    save_animation: bool = True
    save_distance_plot: bool = True

    # Save each rollout as an NPZ file.
    save_rollout_data: bool = True

    # Aggregate outputs written inside output_dir.
    summary_csv: str = "summary.csv"
    summary_npz: str = "summary.npz"

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
    angle_offset: float = 0.0,
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
        + angle_offset
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
    angle_offset: float = 0.0,
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
        angle_offset=angle_offset,
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

def rollout_position_swap_original_dynamics(
    env,
    act_fn,
    key,
    rollout_length: int,
    radius_ratio: float,
    angle_offset: float = 0.0,
    show_progress: bool = False,
):
    """
    Original simulator loop:

        graph_t
          -> policy(graph_t)
          -> env.step(graph_t, action_t)
          -> graph_{t+1}

    No Vicon estimates, no delay, and no added noise.
    """
    graph_true = env.reset(key)

    graph_true, start_xy, goal_xy = set_position_swap_graph(
        env=env,
        graph=graph_true,
        radius_ratio=radius_ratio,
        angle_offset=angle_offset,
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

    step_iterator = (
        trange(rollout_length, ncols=80)
        if show_progress
        else range(rollout_length)
    )

    for _ in step_iterator:
        action = act_fn(graph_true)

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

        per_agent_min_distance_log.append(
            per_agent_min_distance(next_agent_states)
        )

        graph_true = graph_next

        if bool(np.any(np.asarray(done))):
            break

    return {
        "start_xy": np.asarray(start_xy),
        "goal_xy": np.asarray(goal_xy),
        "true_states": np.stack(true_states_log, axis=0),
        "actions": np.stack(actions_log, axis=0),
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
        "GCBF-Diffuser Position Exchange with Dynamic Graph Edges"
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
# MULTI-TRIAL METRICS / REPORTING
# ============================================================

def summarize_rollout(results, dt: float, goal_tolerance: float):
    """Return scalar statistics for one position-swap rollout."""
    finish = np.asarray(results["finish"], dtype=bool)
    unsafe = np.asarray(results["unsafe"], dtype=bool)
    states = np.asarray(results["true_states"])
    goal_xy = np.asarray(results["goal_xy"])

    # finish shape: [T, N]
    agent_ever_finished = np.any(finish, axis=0)
    all_agents_ever_finished = bool(np.all(agent_ever_finished))
    all_agents_simultaneous = bool(np.any(np.all(finish, axis=1)))
    all_agents_final = bool(np.all(finish[-1]))

    first_finish_steps = np.full(finish.shape[1], np.nan, dtype=np.float32)
    for agent_idx in range(finish.shape[1]):
        reached = np.flatnonzero(finish[:, agent_idx])
        if reached.size:
            first_finish_steps[agent_idx] = reached[0] + 1

    first_finish_times = first_finish_steps * dt
    mean_arrival_time = (
        float(np.nanmean(first_finish_times))
        if np.any(np.isfinite(first_finish_times))
        else np.nan
    )
    max_arrival_time = (
        float(np.nanmax(first_finish_times))
        if np.any(np.isfinite(first_finish_times))
        else np.nan
    )

    final_xy = states[-1, :, :2]
    final_goal_errors = np.linalg.norm(final_xy - goal_xy, axis=-1)

    collision_free = not bool(np.any(unsafe))
    collision_free_success = collision_free and all_agents_ever_finished

    return {
        "individual_finish_rate": float(np.mean(agent_ever_finished)),
        "all_agents_ever_finished": all_agents_ever_finished,
        "all_agents_simultaneous": all_agents_simultaneous,
        "all_agents_final": all_agents_final,
        "collision_free": collision_free,
        "collision_free_success": collision_free_success,
        "mean_arrival_time_s": mean_arrival_time,
        "max_arrival_time_s": max_arrival_time,
        "minimum_pairwise_distance_m": float(np.min(results["min_distance"])),
        "mean_final_goal_error_m": float(np.mean(final_goal_errors)),
        "max_final_goal_error_m": float(np.max(final_goal_errors)),
        "maximum_cost": float(np.max(results["costs"])),
        "goal_tolerance_m": float(goal_tolerance),
    }


def print_batch_summary(rows):
    """Print aggregate statistics across all trials."""
    n_trials = len(rows)
    individual_finish = np.asarray([r["individual_finish_rate"] for r in rows])
    all_finish = np.asarray([r["all_agents_ever_finished"] for r in rows], dtype=float)
    simultaneous = np.asarray([r["all_agents_simultaneous"] for r in rows], dtype=float)
    final_finish = np.asarray([r["all_agents_final"] for r in rows], dtype=float)
    collision_free = np.asarray([r["collision_free"] for r in rows], dtype=float)
    safe_success = np.asarray([r["collision_free_success"] for r in rows], dtype=float)
    min_distance = np.asarray([r["minimum_pairwise_distance_m"] for r in rows])
    mean_goal_error = np.asarray([r["mean_final_goal_error_m"] for r in rows])
    max_goal_error = np.asarray([r["max_final_goal_error_m"] for r in rows])
    mean_arrival = np.asarray([r["mean_arrival_time_s"] for r in rows])

    print("\n================ BATCH SUMMARY ================")
    print(f"Trials: {n_trials}")
    print(f"Mean individual-agent finish rate: {100.0 * individual_finish.mean():.2f}%")
    print(f"All-agents finish rate: {100.0 * all_finish.mean():.2f}%")
    print(f"All-agents simultaneous finish rate: {100.0 * simultaneous.mean():.2f}%")
    print(f"All-agents final-step finish rate: {100.0 * final_finish.mean():.2f}%")
    print(f"Collision-free rollout rate: {100.0 * collision_free.mean():.2f}%")
    print(f"Collision-free all-agent success rate: {100.0 * safe_success.mean():.2f}%")
    print(
        "Minimum pairwise distance [mean/median/worst]: "
        f"{min_distance.mean():.3f} / {np.median(min_distance):.3f} / {min_distance.min():.3f} m"
    )
    print(
        "Final goal error mean across trials: "
        f"{mean_goal_error.mean():.3f} m"
    )
    print(
        "Worst-agent final goal error [mean/worst]: "
        f"{max_goal_error.mean():.3f} / {max_goal_error.max():.3f} m"
    )
    if np.any(np.isfinite(mean_arrival)):
        print(f"Mean first-arrival time: {np.nanmean(mean_arrival):.3f} s")
    print("================================================\n")


def save_summary_csv(rows, filename: str):
    fieldnames = list(rows[0].keys())
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary CSV: {filename}")


def save_summary_npz(rows, filename: str):
    arrays = {}
    for key in rows[0].keys():
        arrays[key] = np.asarray([row[key] for row in rows])
    np.savez(filename, **arrays)
    print(f"Saved summary NPZ: {filename}")


# ============================================================
# MAIN TEST
# ============================================================

def test(args: Args):
    print(f"> Running {args.n_trials} position-swap tests: {args}")

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"

    if args.debug:
        jax.config.update("jax_disable_jit", True)

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    config_path = os.path.join(args.path, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)

    env = make_env(
        env_id=config.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        area_size=args.area_size,
        max_step=args.max_step,
        max_travel=args.max_travel,
    )

    model_path = os.path.join(args.path, "models")

    if args.step is None:
        checkpoints = [
            int(name)
            for name in os.listdir(model_path)
            if name.isdigit()
        ]
        if not checkpoints:
            raise FileNotFoundError(
                f"No numeric checkpoint folders found in: {model_path}"
            )
        step = max(checkpoints)
    else:
        step = args.step

    print("Loading model checkpoint:", step)

    algo = make_algo(
        algo=args.algo,
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
    act_fn = jax.jit(algo.act)

    debug_graph = env.reset(jr.PRNGKey(args.seed))
    print("\nGraph diagnostics:")
    print("states shape:", debug_graph.states.shape)
    print("node types:", np.unique(np.asarray(debug_graph.node_type)))
    print("comm_radius:", float(env._params["comm_radius"]))
    print("car_radius:", float(env._params["car_radius"]))

    rng = np.random.default_rng(args.seed)
    trial_rows = []

    for trial_idx in range(args.n_trials):
        trial_seed = args.seed + trial_idx

        angle_offset = (
            float(rng.uniform(0.0, 2.0 * np.pi))
            if args.randomize_angle
            else 0.0
        )

        radius_ratio = args.radius_ratio
        if args.radius_ratio_jitter > 0.0:
            radius_ratio += float(
                rng.uniform(
                    -args.radius_ratio_jitter,
                    args.radius_ratio_jitter,
                )
            )

        # Keep the circular configuration inside the square workspace.
        radius_ratio = float(np.clip(radius_ratio, 0.05, 0.49))

        print(
            f"\n--- Trial {trial_idx + 1:02d}/{args.n_trials:02d} "
            f"| seed={trial_seed} "
            f"| angle={angle_offset:.3f} rad "
            f"| radius_ratio={radius_ratio:.3f} ---"
        )

        results = rollout_position_swap_original_dynamics(
            env=env,
            act_fn=act_fn,
            key=jr.PRNGKey(trial_seed),
            rollout_length=args.max_step,
            radius_ratio=radius_ratio,
            angle_offset=angle_offset,
            show_progress=False,
        )

        row = summarize_rollout(
            results=results,
            dt=float(env.dt),
            goal_tolerance=2.0 * float(env._params["car_radius"]),
        )
        row = {
            "trial": trial_idx,
            "seed": trial_seed,
            "angle_offset_rad": angle_offset,
            "radius_ratio": radius_ratio,
            **row,
        }
        trial_rows.append(row)

        print(
            f"finish={100.0 * row['individual_finish_rate']:.1f}% | "
            f"all_finish={row['all_agents_ever_finished']} | "
            f"unsafe={not row['collision_free']} | "
            f"min_dist={row['minimum_pairwise_distance_m']:.3f} m | "
            f"mean_goal_error={row['mean_final_goal_error_m']:.3f} m"
        )

        stem = f"trial_{trial_idx:02d}_seed_{trial_seed}"

        if args.save_rollout_data:
            results_to_save = {
                key: value
                for key, value in results.items()
                if key != "graph_log"
            }
            np.savez(
                os.path.join(args.output_dir, f"{stem}.npz"),
                **results_to_save,
            )

        if args.save_animation:
            animate_position_swap_graph(
                results=results,
                area_size=args.area_size,
                filename=os.path.join(args.output_dir, f"{stem}.gif"),
                comm_radius=float(env._params["comm_radius"]),
                interval_ms=50,
            )

        if args.save_distance_plot:
            plot_per_agent_min_distance(
                results=results,
                dt=float(env.dt),
                car_radius=float(env._params["car_radius"]),
                filename=os.path.join(
                    args.output_dir,
                    f"{stem}_min_distance.png",
                ),
            )

        # Release the large variable-size graph snapshots before next trial.
        del results

    summary_csv_path = os.path.join(args.output_dir, args.summary_csv)
    summary_npz_path = os.path.join(args.output_dir, args.summary_npz)

    save_summary_csv(trial_rows, summary_csv_path)
    save_summary_npz(trial_rows, summary_npz_path)
    print_batch_summary(trial_rows)

    return trial_rows


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
            "/home/sharma/Projects/gcbfplus/logs/"
            "DoubleIntegrator/gcbf+/seed0_20260728015821"
        ),
        n_trials=25,
        output_dir="crazyflie_position_swap_25_cases",
        save_animation=True,
        save_distance_plot=True,
        save_rollout_data=True,
        radius_ratio=0.35,
        randomize_angle=True,
        radius_ratio_jitter=0.00, #0.02
    )

    trial_rows = test(args)
