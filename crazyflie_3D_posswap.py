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
    env: str = "CrazyFlie"
    algo: str = "gcbf_diffuser"

    num_agents: int = 8
    obs: int = 0
    area_size: float = 4.0
    max_step: int = 400
    max_travel: Optional[float] = None

    path: str = (
        "/home/sharma/Projects/gcbfplus/pretrained_diffuser/"
        "CrazyFlie/gcbfdiffuser"
    )
    step: Optional[int] = None

    seed: int = 1234
    debug: bool = False
    cpu: bool = False

    # 3D position-swap geometry
    radius_ratio: float = 0.25
    z_center: float = 1.0
    z_half_span: float = 0.30

    save_animation: bool = True
    animation_file: str = "crazyflie_3d_position_swap.gif"

    save_distance_plot: bool = True
    distance_plot_file: str = "crazyflie_per_agent_min_clearance.png"

    results_file: str = "crazyflie_3d_position_swap.npz"

    def __post_init__(self):
        assert self.num_agents % 2 == 0
        assert self.z_center - self.z_half_span > 0.0


# ============================================================
# GRAPH HELPERS
# ============================================================

def get_agent_ids(graph: GraphsTuple, num_agents: int):
    return jnp.where(
        graph.node_type == 0,
        size=num_agents,
        fill_value=0,
    )[0]


def get_goal_ids(graph: GraphsTuple, num_agents: int):
    return jnp.where(
        graph.node_type == 1,
        size=num_agents,
        fill_value=0,
    )[0]


def snapshot_graph(graph: GraphsTuple):
    return {
        "states": np.asarray(graph.states).copy(),
        "node_type": np.asarray(graph.node_type).copy(),
        "senders": np.asarray(graph.senders).copy(),
        "receivers": np.asarray(graph.receivers).copy(),
    }


# ============================================================
# 3D POSITION-SWAP INITIALIZATION
# ============================================================

def make_position_swap_configuration_3d(
    num_agents: int,
    area_size: float,
    radius_ratio: float,
    z_center: float,
    z_half_span: float,
):
    """
    Agents 0 ... N/2-1 start on lower layer.
    Agents N/2 ... N-1 start on upper layer.

    Agent i moves to the initial 3D location of agent i + N/2.
    """
    assert num_agents % 2 == 0

    n_pairs = num_agents // 2

    center_xy = jnp.array([
        area_size / 2.0,
        area_size / 2.0,
    ])

    radius_xy = radius_ratio * area_size

    theta_lower = (
        2.0
        * jnp.pi
        * jnp.arange(n_pairs)
        / n_pairs
    )

    # Upper layer is horizontally opposite the lower layer.
    theta_upper = theta_lower + jnp.pi

    lower_xy = center_xy + radius_xy * jnp.stack(
        [
            jnp.cos(theta_lower),
            jnp.sin(theta_lower),
        ],
        axis=1,
    )

    upper_xy = center_xy + radius_xy * jnp.stack(
        [
            jnp.cos(theta_upper),
            jnp.sin(theta_upper),
        ],
        axis=1,
    )

    lower_z = jnp.full(
        (n_pairs, 1),
        z_center - z_half_span,
    )

    upper_z = jnp.full(
        (n_pairs, 1),
        z_center + z_half_span,
    )

    lower_xyz = jnp.concatenate(
        [lower_xy, lower_z],
        axis=1,
    )

    upper_xyz = jnp.concatenate(
        [upper_xy, upper_z],
        axis=1,
    )

    start_xyz = jnp.concatenate(
        [lower_xyz, upper_xyz],
        axis=0,
    )

    goal_xyz = jnp.roll(
        start_xyz,
        shift=n_pairs,
        axis=0,
    )

    return start_xyz, goal_xyz


def set_position_swap_graph(
    env,
    graph: GraphsTuple,
    radius_ratio: float,
    z_center: float,
    z_half_span: float,
):
    """
    CrazyFlie state ordering:

    [x, y, z, psi, theta, phi, u, v, w, r, q, p]
    """
    n_agents = env.num_agents

    start_xyz, goal_xyz = make_position_swap_configuration_3d(
        num_agents=n_agents,
        area_size=env.area_size,
        radius_ratio=radius_ratio,
        z_center=z_center,
        z_half_span=z_half_span,
    )

    agent_states = jnp.zeros(
        (n_agents, env.state_dim),
        dtype=graph.states.dtype,
    )
    agent_states = agent_states.at[:, :3].set(start_xyz)

    goal_states = jnp.zeros(
        (n_agents, env.state_dim),
        dtype=graph.states.dtype,
    )
    goal_states = goal_states.at[:, :3].set(goal_xyz)

    new_env_state = env.EnvState(
        agent=agent_states,
        goal=goal_states,
        obstacle=graph.env_states.obstacle,
    )

    # Important: rebuild the entire graph, including 3D communication edges.
    graph_new = env.get_graph(new_env_state)

    return graph_new, start_xyz, goal_xyz


# ============================================================
# DISTANCE / CLEARANCE METRICS
# ============================================================

def pairwise_min_distance_3d(agent_states: np.ndarray) -> float:
    """Minimum raw 3D center-to-center drone distance."""
    xyz = agent_states[:, :3]
    n_agents = xyz.shape[0]

    min_distance = np.inf

    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            distance = np.linalg.norm(xyz[i] - xyz[j])
            min_distance = min(min_distance, distance)

    return float(min_distance)


def get_obstacle_centers_and_radii(obstacles):
    """
    Extract centers/radii from the project's Sphere representation.

    Handles common attribute names:
      center, pos, position
      radius, radii
    """
    centers = None
    radii = None

    for name in ["center", "centers", "pos", "position"]:
        if hasattr(obstacles, name):
            centers = np.asarray(getattr(obstacles, name))
            break

    for name in ["radius", "radii"]:
        if hasattr(obstacles, name):
            radii = np.asarray(getattr(obstacles, name))
            break

    if centers is None or radii is None:
        return np.empty((0, 3)), np.empty((0,))

    centers = np.asarray(centers).reshape(-1, 3)
    radii = np.asarray(radii).reshape(-1)

    return centers, radii


def per_agent_min_clearance(
    agent_states: np.ndarray,
    obstacles,
    drone_radius: float,
):
    """
    For every agent, return the minimum physical clearance to:

    1. another drone:
          ||p_i - p_j|| - 2 * drone_radius

    2. an obstacle:
          ||p_i - c_obs|| - r_obs - drone_radius

    Returns:
        clearance.shape = (num_agents,)
    """
    xyz = agent_states[:, :3]
    n_agents = xyz.shape[0]

    min_clearance = np.full(n_agents, np.inf)

    # Drone-drone clearances.
    for i in range(n_agents):
        for j in range(n_agents):
            if i == j:
                continue

            center_distance = np.linalg.norm(
                xyz[i] - xyz[j]
            )

            clearance = (
                center_distance
                - 2.0 * drone_radius
            )

            min_clearance[i] = min(
                min_clearance[i],
                clearance,
            )

    # Drone-obstacle clearances.
    obstacle_centers, obstacle_radii = (
        get_obstacle_centers_and_radii(obstacles)
    )

    for center, radius in zip(
        obstacle_centers,
        obstacle_radii,
    ):
        for i in range(n_agents):
            center_distance = np.linalg.norm(
                xyz[i] - center
            )

            clearance = (
                center_distance
                - radius
                - drone_radius
            )

            min_clearance[i] = min(
                min_clearance[i],
                clearance,
            )

    return min_clearance


# ============================================================
# COMMUNICATION EDGES
# ============================================================

def active_agent_edges_3d(
    agent_xyz: np.ndarray,
    comm_radius: float,
):
    """
    Matches CrazyFlie.edge_blocks():

        ||p_i - p_j|| < comm_radius
    """
    n_agents = agent_xyz.shape[0]
    edges = []

    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            distance = np.linalg.norm(
                agent_xyz[i] - agent_xyz[j]
            )

            if distance < comm_radius:
                edges.append((i, j))

    return edges


# ============================================================
# PERFECT-STATE CRAZYFLIE ROLLOUT
# ============================================================

def rollout_crazyflie_position_swap(
    env,
    act_fn,
    key,
    rollout_length: int,
    radius_ratio: float,
    z_center: float,
    z_half_span: float,
):
    """
    Policy action:

        [vx_target, vy_target, vz_target, yaw_rate_target]
    """
    graph_true = env.reset(key)

    graph_true, start_xyz, goal_xyz = set_position_swap_graph(
        env=env,
        graph=graph_true,
        radius_ratio=radius_ratio,
        z_center=z_center,
        z_half_span=z_half_span,
    )

    agent_ids = get_agent_ids(
        graph_true,
        env.num_agents,
    )

    drone_radius = float(
        env._params["drone_radius"]
    )

    initial_agent_states = np.asarray(
        graph_true.states[agent_ids]
    )

    states_log = [initial_agent_states]
    graph_log = [snapshot_graph(graph_true)]

    actions_log = []
    rewards_log = []
    costs_log = []
    done_log = []
    unsafe_log = []
    finish_log = []

    min_distance_log = [
        pairwise_min_distance_3d(
            initial_agent_states
        )
    ]

    per_agent_clearance_log = [
        per_agent_min_clearance(
            initial_agent_states,
            graph_true.env_states.obstacle,
            drone_radius,
        )
    ]

    for _ in trange(rollout_length, ncols=80):
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

        states_log.append(next_agent_states)
        graph_log.append(snapshot_graph(graph_next))

        actions_log.append(np.asarray(action))
        rewards_log.append(np.asarray(reward))
        costs_log.append(np.asarray(cost))
        done_log.append(np.asarray(done))
        unsafe_log.append(np.asarray(unsafe))
        finish_log.append(np.asarray(finish))

        min_distance_log.append(
            pairwise_min_distance_3d(
                next_agent_states
            )
        )

        per_agent_clearance_log.append(
            per_agent_min_clearance(
                next_agent_states,
                graph_next.env_states.obstacle,
                drone_radius,
            )
        )

        graph_true = graph_next

        if bool(np.any(np.asarray(done))):
            break

    return {
        "start_xyz": np.asarray(start_xyz),
        "goal_xyz": np.asarray(goal_xyz),
        "states": np.stack(states_log, axis=0),
        "actions": np.stack(actions_log, axis=0),
        "rewards": np.stack(rewards_log, axis=0),
        "costs": np.stack(costs_log, axis=0),
        "done": np.stack(done_log, axis=0),
        "unsafe": np.stack(unsafe_log, axis=0),
        "finish": np.stack(finish_log, axis=0),
        "min_distance": np.asarray(min_distance_log),
        "per_agent_min_clearance": np.stack(
            per_agent_clearance_log,
            axis=0,
        ),
        "graph_log": graph_log,
    }


# ============================================================
# PER-AGENT MINIMUM-CLEARANCE PLOT
# ============================================================

def plot_per_agent_min_clearance(
    results,
    dt: float,
    filename: str,
):
    """
    One curve per drone.

    y = minimum physical clearance to another drone or obstacle.
    The red dashed line at y=0 means physical contact.
    """
    clearance = results["per_agent_min_clearance"]

    n_steps, n_agents = clearance.shape

    time = np.arange(n_steps) * dt

    fig, ax = plt.subplots(
        figsize=(8, 4.5),
    )

    for i in range(n_agents):
        ax.plot(
            time,
            clearance[:, i],
            linewidth=2.0,
            label=f"{i}",
        )

    ax.axhline(
        0.0,
        color="red",
        linestyle="--",
        linewidth=2.0,
        label="Collision boundary",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Min clearance (m)")
    ax.set_title(
        "Minimum Distance to Other Agents and Obstacles"
    )

    ax.grid(True, alpha=0.30)

    ax.legend(
        title="Agent",
        ncol=2,
        fontsize=9,
        loc="upper right",
    )

    fig.tight_layout()

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved clearance plot: {filename}")


# ============================================================
# 3D ANIMATION
# ============================================================

def animate_crazyflie_position_swap_3d(
    results,
    area_size: float,
    filename: str,
    comm_radius: float,
    interval_ms: int = 50,
):
    states = results["states"]
    true_xyz = states[:, :, :3]

    goal_xyz = results["goal_xyz"]

    n_frames = true_xyz.shape[0]
    n_agents = true_xyz.shape[1]

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    ax.set_xlim(0.0, area_size)
    ax.set_ylim(0.0, area_size)
    ax.set_zlim(0.0, area_size)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

    ax.set_title(
        "CrazyFlie 3D Position Exchange"
    )

    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass

    goal_scatter = ax.scatter(
        goal_xyz[:, 0],
        goal_xyz[:, 1],
        goal_xyz[:, 2],
        s=90,
        c="limegreen",
        edgecolors="forestgreen",
        linewidths=1.2,
        label="Goals",
    )

    agent_scatter = ax.scatter(
        true_xyz[0, :, 0],
        true_xyz[0, :, 1],
        true_xyz[0, :, 2],
        s=100,
        c="dodgerblue",
        edgecolors="navy",
        linewidths=1.2,
        label="Agents",
    )

    labels = []

    for i in range(n_agents):
        label = ax.text(
            true_xyz[0, i, 0],
            true_xyz[0, i, 1],
            true_xyz[0, i, 2],
            str(i),
            fontsize=10,
            fontweight="bold",
        )
        labels.append(label)

    trajectory_lines = []

    for i in range(n_agents):
        line, = ax.plot(
            true_xyz[:1, i, 0],
            true_xyz[:1, i, 1],
            true_xyz[:1, i, 2],
            alpha=0.30,
            linewidth=1.0,
        )
        trajectory_lines.append(line)

    task_lines = []
    comm_lines = []

    status_text = ax.text2D(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.85,
        },
    )

    ax.plot(
        [],
        [],
        [],
        color="limegreen",
        linewidth=2.0,
        label="Agent-goal edges",
    )

    ax.plot(
        [],
        [],
        [],
        color="black",
        linewidth=1.5,
        label="Communication edges",
    )

    ax.legend(loc="lower right")

    def clear_edges():
        nonlocal task_lines
        nonlocal comm_lines

        for line in task_lines:
            line.remove()

        for line in comm_lines:
            line.remove()

        task_lines = []
        comm_lines = []

    def update(frame):
        nonlocal task_lines
        nonlocal comm_lines

        agent_xyz = true_xyz[frame]

        agent_scatter._offsets3d = (
            agent_xyz[:, 0],
            agent_xyz[:, 1],
            agent_xyz[:, 2],
        )

        for i, label in enumerate(labels):
            label.set_position(
                (
                    agent_xyz[i, 0],
                    agent_xyz[i, 1],
                )
            )
            label.set_3d_properties(
                agent_xyz[i, 2],
            )

        for i, line in enumerate(trajectory_lines):
            line.set_data(
                true_xyz[:frame + 1, i, 0],
                true_xyz[:frame + 1, i, 1],
            )
            line.set_3d_properties(
                true_xyz[:frame + 1, i, 2]
            )

        clear_edges()

        # Green task edges.
        for i in range(n_agents):
            p_agent = agent_xyz[i]
            p_goal = goal_xyz[i]

            line, = ax.plot(
                [p_agent[0], p_goal[0]],
                [p_agent[1], p_goal[1]],
                [p_agent[2], p_goal[2]],
                color="limegreen",
                alpha=0.45,
                linewidth=1.8,
            )

            task_lines.append(line)

        # Black communication edges.
        comm_edges = active_agent_edges_3d(
            agent_xyz,
            comm_radius,
        )

        for i, j in comm_edges:
            p_i = agent_xyz[i]
            p_j = agent_xyz[j]

            line, = ax.plot(
                [p_i[0], p_j[0]],
                [p_i[1], p_j[1]],
                [p_i[2], p_j[2]],
                color="black",
                alpha=0.70,
                linewidth=1.4,
            )

            comm_lines.append(line)

        if frame == 0:
            reward = 0.0
            cost = 0.0
            unsafe = False
        else:
            reward = results["rewards"][frame - 1]
            cost = results["costs"][frame - 1]
            unsafe = results["unsafe"][frame - 1]

        status_text.set_text(
            f"Cost: {float(np.asarray(cost)):.4f}\n"
            f"Reward: {float(np.asarray(reward)):.4f}\n"
            f"Unsafe: {bool(np.any(np.asarray(unsafe)))}\n"
            f"Step: {frame:04d}\n"
            f"Min pair distance: "
            f"{results['min_distance'][frame]:.3f} m"
        )

        return [
            agent_scatter,
            goal_scatter,
            status_text,
            *labels,
            *trajectory_lines,
            *task_lines,
            *comm_lines,
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
    print(f"> Running CrazyFlie 3D position swap: {args}")

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
        env_id=args.env,
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

        if not checkpoints:
            raise FileNotFoundError(
                f"No numeric checkpoint folders in: {model_path}"
            )

        step = max(checkpoints)
    else:
        step = args.step

    print("Loading checkpoint:", step)

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

    print("\nEnvironment diagnostics:")
    print("state_dim:", env.state_dim)
    print("action_dim:", env.action_dim)
    print("comm_radius:", float(env.comm_radius))
    print("drone_radius:", float(env._params["drone_radius"]))

    results = rollout_crazyflie_position_swap(
        env=env,
        act_fn=act_fn,
        key=jr.PRNGKey(args.seed),
        rollout_length=args.max_step,
        radius_ratio=args.radius_ratio,
        z_center=args.z_center,
        z_half_span=args.z_half_span,
    )

    print("\n========== RESULTS ==========")
    print("State trajectory:", results["states"].shape)
    print("Action trajectory:", results["actions"].shape)
    print("Unsafe:", bool(np.any(results["unsafe"])))
    print("Goal reached:", bool(np.any(results["finish"])))
    print("Maximum cost:", float(np.max(results["costs"])))
    print(
        "Minimum pairwise distance:",
        float(np.min(results["min_distance"])),
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
        animate_crazyflie_position_swap_3d(
            results=results,
            area_size=args.area_size,
            filename=args.animation_file,
            comm_radius=float(env.comm_radius),
            interval_ms=50,
        )

    if args.save_distance_plot:
        plot_per_agent_min_clearance(
            results=results,
            dt=float(env.dt),
            filename=args.distance_plot_file,
        )

    return results


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    args = Args(
        env="CrazyFlie",
        algo="gcbf_diffuser",
        num_agents=8,
        obs=0,
        area_size=2.0,
        max_step=400,
        path=(
            "/home/sharma/Projects/gcbfplus/pretrained_diffuser/"
            "CrazyFlie/gcbfdiffuser"
        ),
        radius_ratio=0.25,
        z_center=1.0,
        z_half_span=0.30,
        save_animation=True,
        save_distance_plot=True,
        animation_file="crazyflie_3d_position_swap.gif",
        distance_plot_file="crazyflie_per_agent_min_clearance.png",
        results_file="crazyflie_3d_position_swap.npz",
    )

    results = test(args)