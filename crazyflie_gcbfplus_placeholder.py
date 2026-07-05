#!/usr/bin/env python3

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

from crazyflie_py import Crazyswarm

from gcbfplus.algo import make_algo
from gcbfplus.env import make_env
from gcbfplus.utils.graph import GraphsTuple


# ============================================================
# USER CONFIGURATION
# ============================================================

@dataclass
class Args:
    # --------------------------------------------------------
    # GCBF+ model/environment
    # --------------------------------------------------------
    env: str = "DoubleIntegrator"
    algo: str = "gcbf_plus"

    num_agents: int = 8
    obs: int = 0
    area_size: float = 4.0
    max_step: int = 1000
    max_travel: Optional[float] = None

    model_path: str = (
        "/home/sharma/Projects/gcbfplus/pretrained/"
        "DoubleIntegrator/gcbf+"
    )
    model_step: Optional[int] = None

    seed: int = 1234
    debug: bool = False
    cpu: bool = False

    # --------------------------------------------------------
    # Position exchange task
    # --------------------------------------------------------
    circle_radius_ratio: float = 0.30

    # --------------------------------------------------------
    # Crazyflie execution
    # --------------------------------------------------------
    Z_REF: float = 0.80
    TAKEOFF_DURATION: float = 3.0
    MOVE_TO_START_DURATION: float = 4.0

    CONTROL_RATE_HZ: float = 50.0
    CMD_HORIZON: float = 0.05

    YAW_REF: float = 0.0

    # Velocity estimation from Vicon positions.
    VELOCITY_FILTER_ALPHA: float = 0.35

    # --------------------------------------------------------
    # Hardware safety limits
    # --------------------------------------------------------
    MIN_SAFE_DISTANCE: float = 0.22
    MAX_ALTITUDE_ERROR: float = 0.15
    GOAL_TOLERANCE: float = 0.10

    # Conservative acceleration clipping.
    MAX_PLANAR_ACCELERATION: float = 1.5

    # --------------------------------------------------------
    # Logging / animation
    # --------------------------------------------------------
    SAVE_ANIMATION: bool = True
    ANIMATION_FILE: str = "crazyflie_position_swap.gif"
    RESULTS_FILE: str = "crazyflie_position_swap_log.npz"

    def __post_init__(self):
        assert self.num_agents % 2 == 0, (
            "Position swapping requires an even number of agents."
        )


# ============================================================
# LOAD GCBF+ MODEL
# ============================================================

def load_gcbf_controller(args: Args):
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"

    if args.debug:
        jax.config.update("jax_disable_jit", True)

    np.random.seed(args.seed)

    config_path = os.path.join(
        args.model_path,
        "config.yaml",
    )

    with open(config_path, "r") as f:
        config = yaml.load(
            f,
            Loader=yaml.UnsafeLoader,
        )

    env = make_env(
        env_id=config.env,
        num_agents=args.num_agents,
        num_obs=args.obs,
        area_size=args.area_size,
        max_step=args.max_step,
        max_travel=args.max_travel,
    )

    checkpoint_dir = os.path.join(
        args.model_path,
        "models",
    )

    if args.model_step is None:
        available_steps = [
            int(name)
            for name in os.listdir(checkpoint_dir)
            if name.isdigit()
        ]

        if len(available_steps) == 0:
            raise FileNotFoundError(
                f"No checkpoint found in {checkpoint_dir}"
            )

        model_step = max(available_steps)
    else:
        model_step = args.model_step

    print(f"Loading GCBF+ checkpoint: {model_step}")

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

    algo.load(checkpoint_dir, model_step)

    # Explicit parameters avoid hidden dependence on mutable train state.
    params = algo.actor_train_state.params

    act_fn = jax.jit(algo.act)

    return env, algo, params, act_fn


# ============================================================
# GRAPH HELPERS
# ============================================================

def get_agent_ids(graph: GraphsTuple, n_agents: int):
    """
    GCBF+ convention:
        node_type == 0 -> agent nodes.
    """
    return jnp.where(
        graph.node_type == 0,
        size=n_agents,
        fill_value=0,
    )[0]


def get_goal_ids(graph: GraphsTuple, n_agents: int):
    """
    Expected convention:
        node_type == 1 -> goal nodes.

    Change this only if your printed node types show another
    value for goal nodes.
    """
    return jnp.where(
        graph.node_type == 1,
        size=n_agents,
        fill_value=0,
    )[0]


def make_position_swap_configuration(
    n_agents: int,
    area_size: float,
    radius_ratio: float,
):
    """
    Drone i starts on a circle and targets the start position
    of drone i + N/2.
    """
    center = jnp.array([
        area_size / 2.0,
        area_size / 2.0,
    ])

    radius = radius_ratio * area_size

    theta = (
        2.0
        * jnp.pi
        * jnp.arange(n_agents)
        / n_agents
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
        shift=n_agents // 2,
        axis=0,
    )

    return start_xy, goal_xy


def initialize_position_swap_graph(
    env,
    key,
    radius_ratio: float,
):
    """
    Creates one graph template with fixed circular starts/goals.

    On hardware, the agent states will later be replaced by
    Vicon-derived [x, y, vx, vy] estimates every iteration.
    """
    graph = env.reset(key)

    n_agents = env.num_agents

    agent_ids = get_agent_ids(graph, n_agents)
    goal_ids = get_goal_ids(graph, n_agents)

    start_xy, goal_xy = make_position_swap_configuration(
        n_agents=n_agents,
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

    all_states = graph.states.at[agent_ids].set(agent_states)
    all_states = all_states.at[goal_ids].set(goal_states)

    graph = env.add_edge_feats(graph, all_states)

    return graph, start_xy, goal_xy


def graph_from_measured_states(
    env,
    graph_template: GraphsTuple,
    measured_states_2d: np.ndarray,
):
    """
    Replace only the agent nodes using physical measurements.

    measured_states_2d:
        shape (N, 4)
        [x, y, vx, vy]
    """
    agent_ids = get_agent_ids(
        graph_template,
        env.num_agents,
    )

    new_states = graph_template.states.at[agent_ids].set(
        jnp.asarray(measured_states_2d)
    )

    return env.add_edge_feats(
        graph_template,
        new_states,
    )


# ============================================================
# VICON POSITION / VELOCITY ESTIMATOR
# ============================================================

class ViconStateEstimator:
    """
    `cf.position()` is used as the measured position source.

    Velocity is estimated by filtered finite differences:

        v_raw = (p_k - p_{k-1}) / dt
        v_hat = alpha * v_raw + (1-alpha) * v_hat_previous
    """

    def __init__(
        self,
        n_agents: int,
        velocity_filter_alpha: float,
    ):
        self.n_agents = n_agents
        self.alpha = velocity_filter_alpha

        self.prev_position = None
        self.prev_time = None
        self.velocity = np.zeros((n_agents, 3))

    def reset(self):
        self.prev_position = None
        self.prev_time = None
        self.velocity[:] = 0.0

    def update(self, positions_3d: np.ndarray, current_time: float):
        positions_3d = np.asarray(positions_3d)

        if self.prev_position is None:
            self.prev_position = positions_3d.copy()
            self.prev_time = current_time
            return positions_3d, self.velocity.copy()

        dt = current_time - self.prev_time

        if dt <= 1e-5:
            return positions_3d, self.velocity.copy()

        raw_velocity = (
            positions_3d - self.prev_position
        ) / dt

        self.velocity = (
            self.alpha * raw_velocity
            + (1.0 - self.alpha) * self.velocity
        )

        self.prev_position = positions_3d.copy()
        self.prev_time = current_time

        return positions_3d, self.velocity.copy()


def read_crazyflie_states(
    cfs,
    estimator: ViconStateEstimator,
    current_time: float,
):
    """
    Read 3D position estimates from Crazyswarm and estimate velocity.

    Output:
        pos_3d: (N, 3)
        vel_3d: (N, 3)
        states_2d: (N, 4) = [x, y, vx, vy]
    """
    positions_3d = np.asarray(
        [
            np.asarray(cf.position())
            for cf in cfs
        ]
    )

    pos_3d, vel_3d = estimator.update(
        positions_3d,
        current_time,
    )

    states_2d = np.column_stack(
        [
            pos_3d[:, 0],
            pos_3d[:, 1],
            vel_3d[:, 0],
            vel_3d[:, 1],
        ]
    )

    return pos_3d, vel_3d, states_2d


# ============================================================
# CONTROL HELPERS
# ============================================================

def rollout_double_integrator_reference(
    states_2d: np.ndarray,
    acceleration_xy: np.ndarray,
    dt: float,
):
    """
    Short-horizon ideal rollout used only to construct
    cmdFullState references.

    p_cmd = p + v dt + 0.5 a dt²
    v_cmd = v + a dt
    """
    pos_xy = states_2d[:, 0:2]
    vel_xy = states_2d[:, 2:4]

    pos_xy_cmd = (
        pos_xy
        + vel_xy * dt
        + 0.5 * acceleration_xy * dt**2
    )

    vel_xy_cmd = vel_xy + acceleration_xy * dt

    return pos_xy_cmd, vel_xy_cmd


def clip_planar_acceleration(
    acc_xy: np.ndarray,
    max_acceleration: float,
):
    """
    Limits the norm of each planar acceleration command.
    """
    norms = np.linalg.norm(acc_xy, axis=1, keepdims=True)

    scale = np.minimum(
        1.0,
        max_acceleration / np.maximum(norms, 1e-8),
    )

    return acc_xy * scale


def pairwise_min_distance(states_2d: np.ndarray):
    xy = states_2d[:, :2]
    n_agents = xy.shape[0]

    min_distance = np.inf

    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            distance = np.linalg.norm(
                xy[i] - xy[j]
            )
            min_distance = min(
                min_distance,
                distance,
            )

    return float(min_distance)


def all_agents_at_goals(
    pos_xy: np.ndarray,
    goal_xy: np.ndarray,
    tolerance: float,
):
    errors = np.linalg.norm(
        pos_xy - goal_xy,
        axis=1,
    )

    return bool(np.all(errors <= tolerance))


# ============================================================
# GRAPH SNAPSHOT / ANIMATION
# ============================================================

def snapshot_graph(graph: GraphsTuple):
    return {
        "states": np.asarray(graph.states).copy(),
        "node_type": np.asarray(graph.node_type).copy(),
        "senders": np.asarray(graph.senders).copy(),
        "receivers": np.asarray(graph.receivers).copy(),
    }


def unique_undirected_edges(senders, receivers):
    edges = set()

    for sender, receiver in zip(senders, receivers):
        sender = int(sender)
        receiver = int(receiver)

        if sender != receiver:
            edges.add(tuple(sorted((sender, receiver))))

    return list(edges)


def split_graph_edges(graph_snapshot):
    """
    Green:
        agent-goal edges.

    Black:
        agent-agent edges.
    """
    node_type = graph_snapshot["node_type"]

    green_edges = []
    black_edges = []

    for sender, receiver in unique_undirected_edges(
        graph_snapshot["senders"],
        graph_snapshot["receivers"],
    ):
        sender_type = node_type[sender]
        receiver_type = node_type[receiver]

        if (
            (sender_type == 0 and receiver_type == 1)
            or (sender_type == 1 and receiver_type == 0)
        ):
            green_edges.append((sender, receiver))

        elif sender_type == 0 and receiver_type == 0:
            black_edges.append((sender, receiver))

    return green_edges, black_edges


def animate_hardware_position_swap(
    results,
    area_size: float,
    filename: str,
    interval_ms: int = 50,
):
    graph_log = results["graph_log"]
    state_log = results["states_2d"]

    n_frames = len(graph_log)
    n_agents = state_log.shape[1]

    first_graph = graph_log[0]

    goal_ids = np.where(
        first_graph["node_type"] == 1
    )[0]

    goal_xy = first_graph["states"][goal_ids, :2]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(0.0, area_size)
    ax.set_ylim(0.0, area_size)
    ax.set_aspect("equal")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Crazyflie GCBF+ Position Exchange")

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
        state_log[0, :, 0],
        state_log[0, :, 1],
        s=180,
        c="dodgerblue",
        edgecolors="navy",
        linewidths=1.5,
        zorder=5,
        label="Crazyflies",
    )

    labels = []

    for i in range(n_agents):
        label = ax.text(
            state_log[0, i, 0],
            state_log[0, i, 1],
            str(i),
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
        )
        labels.append(label)

    trajectory_lines = []

    for i in range(n_agents):
        line, = ax.plot(
            state_log[:1, i, 0],
            state_log[:1, i, 1],
            color="dodgerblue",
            alpha=0.35,
            linewidth=1.0,
            zorder=1,
        )
        trajectory_lines.append(line)

    green_lines = []
    black_lines = []

    status_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        verticalalignment="top",
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
        label="Goal edges",
    )

    ax.plot(
        [],
        [],
        color="black",
        linewidth=1.5,
        label="Proximity edges",
    )

    ax.legend(loc="lower right")

    def remove_edges():
        nonlocal green_lines
        nonlocal black_lines

        for line in green_lines:
            line.remove()

        for line in black_lines:
            line.remove()

        green_lines = []
        black_lines = []

    def update(frame):
        nonlocal green_lines
        nonlocal black_lines

        graph = graph_log[frame]
        states = graph["states"]
        node_type = graph["node_type"]

        agent_ids = np.where(node_type == 0)[0]
        agent_xy = states[agent_ids, :2]

        agent_scatter.set_offsets(agent_xy)

        for i, label in enumerate(labels):
            label.set_position(
                (
                    agent_xy[i, 0],
                    agent_xy[i, 1],
                )
            )

        for i, line in enumerate(trajectory_lines):
            line.set_data(
                state_log[:frame + 1, i, 0],
                state_log[:frame + 1, i, 1],
            )

        remove_edges()

        green_edges, black_edges = split_graph_edges(
            graph
        )

        for sender, receiver in green_edges:
            p0 = states[sender, :2]
            p1 = states[receiver, :2]

            line, = ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color="limegreen",
                alpha=0.50,
                linewidth=2.0,
                zorder=2,
            )
            green_lines.append(line)

        for sender, receiver in black_edges:
            p0 = states[sender, :2]
            p1 = states[receiver, :2]

            line, = ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color="black",
                alpha=0.70,
                linewidth=1.5,
                zorder=4,
            )
            black_lines.append(line)

        status_text.set_text(
            f"kk={frame:04d}\n"
            f"min dist: "
            f"{results['min_distance'][frame]:.3f} m\n"
            f"max |z-zref|: "
            f"{results['max_z_error'][frame]:.3f} m"
        )

        return [
            agent_scatter,
            status_text,
            *labels,
            *trajectory_lines,
            *green_lines,
            *black_lines,
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

    print(f"Saved animation: {filename}")


# ============================================================
# HARDWARE EXECUTION
# ============================================================

def run_hardware_position_swap(
    args: Args,
    env,
    params,
    act_fn,
):
    """
    Main hardware control loop.

    The real-world replacement for env.step(...) is:

        send cmdFullState
        -> physical Crazyflie motion
        -> Vicon/Crazyswarm position measurement
        -> new graph
    """

    swarm = Crazyswarm()

    time_helper = swarm.timeHelper
    allcfs = swarm.allcfs

    cfs = allcfs.crazyflies

    if len(cfs) != args.num_agents:
        raise RuntimeError(
            f"Expected {args.num_agents} Crazyflies, "
            f"but Crazyswarm reports {len(cfs)}."
        )

    # Sort by ID to keep graph-agent order fixed.
    cfs = sorted(
        cfs,
        key=lambda cf: cf.id,
    )

    print("Crazyflie IDs:", [cf.id for cf in cfs])

    graph_template, start_xy, goal_xy = (
        initialize_position_swap_graph(
            env=env,
            key=jr.PRNGKey(args.seed),
            radius_ratio=args.circle_radius_ratio,
        )
    )

    print("\nStart XY:")
    print(np.asarray(start_xy))

    print("\nGoal XY:")
    print(np.asarray(goal_xy))

    estimator = ViconStateEstimator(
        n_agents=args.num_agents,
        velocity_filter_alpha=args.VELOCITY_FILTER_ALPHA,
    )

    # --------------------------------------------------------
    # TAKE OFF
    # --------------------------------------------------------
    print("\nTaking off...")

    allcfs.takeoff(
        targetHeight=args.Z_REF,
        duration=args.TAKEOFF_DURATION,
    )

    time_helper.sleep(
        args.TAKEOFF_DURATION + 1.0
    )

    # --------------------------------------------------------
    # MOVE TO INITIAL CIRCLE
    # --------------------------------------------------------
    print("Moving to circular start positions...")

    for i, cf in enumerate(cfs):
        cf.goTo(
            goal=np.array([
                start_xy[i, 0],
                start_xy[i, 1],
                args.Z_REF,
            ]),
            yaw=args.YAW_REF,
            duration=args.MOVE_TO_START_DURATION,
        )

    time_helper.sleep(
        args.MOVE_TO_START_DURATION + 1.0
    )

    # Initialize velocity estimator after start-position motion.
    estimator.reset()

    initial_positions = np.asarray(
        [np.asarray(cf.position()) for cf in cfs]
    )

    estimator.update(
        initial_positions,
        time_helper.time(),
    )

    print("Beginning streaming GCBF+ control.")

    # --------------------------------------------------------
    # LOGS
    # --------------------------------------------------------
    states_log = []
    actions_log = []
    graph_log = []
    min_distance_log = []
    max_z_error_log = []
    time_log = []

    control_rate = args.CONTROL_RATE_HZ

    abort_reason = None
    success = False

    try:
        for kk in trange(
            args.max_step,
            ncols=80,
        ):
            loop_start = time_helper.time()

            # ------------------------------------------------
            # 1. Obtain physical state estimates
            # ------------------------------------------------
            pos_3d, vel_3d, states_2d = read_crazyflie_states(
                cfs=cfs,
                estimator=estimator,
                current_time=loop_start,
            )

            # ------------------------------------------------
            # 2. Hardware safety checks before actuation
            # ------------------------------------------------
            min_distance = pairwise_min_distance(
                states_2d
            )

            max_z_error = float(
                np.max(
                    np.abs(
                        pos_3d[:, 2] - args.Z_REF
                    )
                )
            )

            if min_distance < args.MIN_SAFE_DISTANCE:
                abort_reason = (
                    f"Minimum separation {min_distance:.3f} m "
                    f"< {args.MIN_SAFE_DISTANCE:.3f} m"
                )
                break

            if max_z_error > args.MAX_ALTITUDE_ERROR:
                abort_reason = (
                    f"Altitude deviation {max_z_error:.3f} m "
                    f"> {args.MAX_ALTITUDE_ERROR:.3f} m"
                )
                break

            if all_agents_at_goals(
                pos_xy=states_2d[:, :2],
                goal_xy=np.asarray(goal_xy),
                tolerance=args.GOAL_TOLERANCE,
            ):
                success = True
                break

            # ------------------------------------------------
            # 3. Build graph from Vicon-derived state estimate
            # ------------------------------------------------
            graph = graph_from_measured_states(
                env=env,
                graph_template=graph_template,
                measured_states_2d=states_2d,
            )

            # ------------------------------------------------
            # 4. GCBF+ acceleration command [ax, ay]
            # ------------------------------------------------
            acc_xy = np.asarray(
                act_fn(graph, params)
            )

            acc_xy = clip_planar_acceleration(
                acc_xy,
                args.MAX_PLANAR_ACCELERATION,
            )

            # ------------------------------------------------
            # 5. Ideal double-integrator rollout to produce
            #    cmdFullState position and velocity references
            # ------------------------------------------------
            pos_xy_cmd, vel_xy_cmd = (
                rollout_double_integrator_reference(
                    states_2d=states_2d,
                    acceleration_xy=acc_xy,
                    dt=args.CMD_HORIZON,
                )
            )

            # ------------------------------------------------
            # 6. Send streaming commands
            # ------------------------------------------------
            for i, cf in enumerate(cfs):
                pos_cmd = np.array([
                    pos_xy_cmd[i, 0],
                    pos_xy_cmd[i, 1],
                    args.Z_REF,
                ])

                vel_cmd = np.array([
                    vel_xy_cmd[i, 0],
                    vel_xy_cmd[i, 1],
                    0.0,
                ])

                acc_cmd = np.array([
                    acc_xy[i, 0],
                    acc_xy[i, 1],
                    0.0,
                ])

                cf.cmdFullState(
                    pos_cmd,
                    vel_cmd,
                    acc_cmd,
                    args.YAW_REF,
                    np.zeros(3),
                )

            # ------------------------------------------------
            # 7. Logging
            # ------------------------------------------------
            states_log.append(states_2d.copy())
            actions_log.append(acc_xy.copy())
            graph_log.append(snapshot_graph(graph))
            min_distance_log.append(min_distance)
            max_z_error_log.append(max_z_error)
            time_log.append(loop_start)

            # Keep actual outer-loop frequency.
            time_helper.sleepForRate(
                control_rate
            )

    except KeyboardInterrupt:
        abort_reason = "KeyboardInterrupt"

    finally:
        # Stop streaming setpoints before returning to land().
        for cf in cfs:
            cf.notifySetpointsStop()

        time_helper.sleep(0.2)

        print("Landing all Crazyflies...")

        allcfs.land(
            targetHeight=0.04,
            duration=3.0,
        )

        time_helper.sleep(4.0)

    if success:
        print("SUCCESS: all agents reached their goals.")
    elif abort_reason is not None:
        print(f"ABORTED: {abort_reason}")
    else:
        print("STOPPED: maximum control horizon reached.")

    if len(states_log) == 0:
        raise RuntimeError(
            "No control samples were logged."
        )

    return {
        "states_2d": np.stack(states_log, axis=0),
        "actions": np.stack(actions_log, axis=0),
        "min_distance": np.asarray(min_distance_log),
        "max_z_error": np.asarray(max_z_error_log),
        "time": np.asarray(time_log),
        "goal_xy": np.asarray(goal_xy),
        "start_xy": np.asarray(start_xy),
        "graph_log": graph_log,
        "success": np.asarray(success),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    args = Args(
        env="DoubleIntegrator",
        algo="gcbf_plus",
        num_agents=8,
        obs=0,
        area_size=4.0,
        max_step=1000,
        model_path=(
            "/home/sharma/Projects/gcbfplus/pretrained/"
            "DoubleIntegrator/gcbf+"
        ),

        # Start conservatively.
        Z_REF=0.80,
        CONTROL_RATE_HZ=50.0,
        CMD_HORIZON=0.05,

        MIN_SAFE_DISTANCE=0.22,
        MAX_ALTITUDE_ERROR=0.15,
        GOAL_TOLERANCE=0.10,
        MAX_PLANAR_ACCELERATION=1.5,

        circle_radius_ratio=0.30,

        SAVE_ANIMATION=True,
        ANIMATION_FILE="crazyflie_position_swap.gif",
        RESULTS_FILE="crazyflie_position_swap_log.npz",
    )

    env, algo, params, act_fn = load_gcbf_controller(
        args
    )

    graph_debug = env.reset(
        jr.PRNGKey(args.seed)
    )

    print("\nGraph diagnostics:")
    print("states shape:", graph_debug.states.shape)
    print("node types:", np.asarray(graph_debug.node_type))
    print(
        "unique node types:",
        np.unique(
            np.asarray(graph_debug.node_type)
        ),
    )

    results = run_hardware_position_swap(
        args=args,
        env=env,
        params=params,
        act_fn=act_fn,
    )

    # Do not save graph_log because its edge count can vary by time.
    np.savez(
        args.RESULTS_FILE,
        states_2d=results["states_2d"],
        actions=results["actions"],
        min_distance=results["min_distance"],
        max_z_error=results["max_z_error"],
        time=results["time"],
        goal_xy=results["goal_xy"],
        start_xy=results["start_xy"],
        success=results["success"],
    )

    print(
        f"Saved hardware log: {args.RESULTS_FILE}"
    )

    if args.SAVE_ANIMATION:
        animate_hardware_position_swap(
            results=results,
            area_size=args.area_size,
            filename=args.ANIMATION_FILE,
            interval_ms=int(
                1000.0 / args.CONTROL_RATE_HZ
            ),
        )


if __name__ == "__main__":
    main()