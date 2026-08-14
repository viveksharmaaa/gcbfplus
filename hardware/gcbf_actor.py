import threading
import time
import heapq
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import rclpy
from crazyflie_interfaces.msg import FullState, Status
from crazyflie_interfaces.srv import Arm, NotifySetpointsStop
from motion_capture_tracking_interfaces.msg import NamedPoseArray
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from tf2_msgs.msg import TFMessage

from gcbfplus.algo.centralized_cbf import CentralizedCBF
from gcbfplus.crazyswarm_double_integrator import (
    DEFAULT_CAR_RADIUS,
    DEFAULT_GOAL_TOLERANCE,
    DEFAULT_HOVER_HEIGHT,
    DEFAULT_MAX_RUNTIME,
    DEFAULT_MASS,
    DEFAULT_RATE_HZ,
    LAND_DURATION,
    LAND_HEIGHT,
    N_AGENTS,
    POSE_TIMEOUT,
    TAKEOFF_DURATION,
    default_crazyflies_yaml,
    default_model_dir,
    load_crazyflies,
    make_algo_from_checkpoint,
    make_graph,
    pairwise_min_distance,
)
from gcbfplus.env import make_env
from gcbfplus.gcbf_state_bridge import STATE_WIDTH


GOAL_HOLD_SECONDS = 2.0
GOAL_SETTLING_SECONDS = 0.5
POSE_LOSS_HOLD_SECONDS = 0.25
DEFAULT_LOOKAHEAD_DT = 0.05
DEFAULT_INITIAL_POSITION_TOLERANCE = 0.20
DEFAULT_MIN_RADIO_RSSI = 30
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def red_text(message):
    return f"{RED}{message}{RESET}"


def yellow_text(message):
    return f"{YELLOW}{message}{RESET}"


@dataclass(frozen=True)
class StateSnapshot:
    sequence: int
    source_time: float
    received_time: float
    state: np.ndarray


class GcbfActor(Node):
    def __init__(self):
        super().__init__("gcbf_actor")
        self.declare_parameter("mode", "sim")
        self.declare_parameter("crazyflies_yaml", str(default_crazyflies_yaml()))
        self.declare_parameter("model_dir", "")
        self.declare_parameter("controller", "gcbf")
        self.declare_parameter("step", 1000)
        self.declare_parameter("rate_hz", DEFAULT_RATE_HZ)
        self.declare_parameter("lookahead_dt", DEFAULT_LOOKAHEAD_DT)
        self.declare_parameter("position_ref_accel_scale", 1.0)
        self.declare_parameter("max_planar_acceleration", 0.0)
        self.declare_parameter("dynamic_reference_generator", False)
        self.declare_parameter("max_planar_speed", 0.0)
        self.declare_parameter("max_planar_jerk", 0.0)
        self.declare_parameter("sim_command_delay", 0.0)
        self.declare_parameter("sim_command_delay_jitter", 0.0)
        self.declare_parameter("sim_position_noise_std", 0.0)
        self.declare_parameter("sim_noise_seed", 2026)
        self.declare_parameter("hover_height", DEFAULT_HOVER_HEIGHT)
        self.declare_parameter("hover_epsilon", 0.06)
        self.declare_parameter("takeoff_duration", TAKEOFF_DURATION)
        self.declare_parameter("takeoff_timeout", 60.0)
        self.declare_parameter("landing_disarm_height", 0.0)
        self.declare_parameter("goal_tolerance", DEFAULT_GOAL_TOLERANCE)
        self.declare_parameter("goal_assignment", "opposite")
        self.declare_parameter("max_runtime", DEFAULT_MAX_RUNTIME)
        self.declare_parameter("policy_test_duration", 0.0)
        self.declare_parameter("pose_timeout", POSE_TIMEOUT)
        self.declare_parameter("pose_loss_timeout", 0.10)
        self.declare_parameter("max_action_age", 0.02)
        self.declare_parameter("velocity_filter_window", 0.0)
        self.declare_parameter("velocity_filter_min_samples", 3)
        self.declare_parameter("area_size", 4.0)
        self.declare_parameter("car_radius", DEFAULT_CAR_RADIUS)
        self.declare_parameter("collision_stop_distance", -1.0)
        self.declare_parameter("mass", DEFAULT_MASS)
        self.declare_parameter("policy_velocity_bound", -1.0)
        self.declare_parameter("qp_car_radius", -1.0)
        self.declare_parameter("qp_acceleration_bound", 2.5)
        self.declare_parameter("qp_velocity_bound", 1.0)
        self.declare_parameter("qp_communication_radius", 0.8)
        self.declare_parameter("qp_alpha", 1.0)
        self.declare_parameter("qp_relaxation_penalty", 1000.0)
        self.declare_parameter("seed", 2)
        self.declare_parameter("print_latency", False)
        self.declare_parameter("policy_diagnostics", True)
        self.declare_parameter("initial_position_tolerance", DEFAULT_INITIAL_POSITION_TOLERANCE)
        self.declare_parameter("min_radio_rssi", DEFAULT_MIN_RADIO_RSSI)

        self.mode = str(self.get_parameter("mode").value).lower()
        if self.mode not in {"sim", "real"}:
            raise ValueError("mode must be 'sim' or 'real'")
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        if self.rate_hz <= 0.0:
            raise ValueError("rate_hz must be positive")
        self.min_period = 1.0 / self.rate_hz

        max_runtime = float(self.get_parameter("max_runtime").value)
        self.controller = str(self.get_parameter("controller").value).lower()
        if self.controller == "centralized_qp":
            qp_car_radius = float(self.get_parameter("qp_car_radius").value)
            if qp_car_radius <= 0.0:
                qp_car_radius = float(self.get_parameter("car_radius").value)
            self.env = make_env(
                env_id="DoubleIntegrator",
                num_agents=N_AGENTS,
                num_obs=0,
                area_size=float(self.get_parameter("area_size").value),
                max_step=max(1, int(max_runtime / 0.03)),
                max_travel=None,
                params={
                    "car_radius": qp_car_radius,
                    "m": float(self.get_parameter("mass").value),
                    "max_acceleration": float(
                        self.get_parameter("qp_acceleration_bound").value
                    ),
                    "max_velocity": float(
                        self.get_parameter("qp_velocity_bound").value
                    ),
                    "comm_radius": float(
                        self.get_parameter("qp_communication_radius").value
                    ),
                },
            )
            self.algo = CentralizedCBF(
                env=self.env,
                node_dim=self.env.node_dim,
                edge_dim=self.env.edge_dim,
                state_dim=self.env.state_dim,
                action_dim=self.env.action_dim,
                n_agents=self.env.num_agents,
                alpha=float(self.get_parameter("qp_alpha").value),
            )
        elif self.controller == "gcbf":
            configured_model_dir = str(self.get_parameter("model_dir").value).strip()
            model_dir = (
                Path(configured_model_dir)
                if configured_model_dir
                else default_model_dir()
            )
            self.env, self.algo = make_algo_from_checkpoint(
                model_dir,
                int(self.get_parameter("step").value),
                float(self.get_parameter("area_size").value),
                max_runtime,
                float(self.get_parameter("car_radius").value),
                float(self.get_parameter("mass").value),
                max_velocity=(
                    None
                    if float(self.get_parameter("policy_velocity_bound").value) < 0.0
                    else float(self.get_parameter("policy_velocity_bound").value)
                ),
            )
        else:
            raise ValueError("controller must be 'gcbf' or 'centralized_qp'")
        initial_graph = self.env.reset(jr.PRNGKey(int(self.get_parameter("seed").value)))
        self.obstacle_state = initial_graph.env_states.obstacle
        self.mass = float(self.env._params["m"])
        # The independent deployment safeguard represents the physical collision
        # boundary. A centralized QP may use a larger radius as planning margin.
        collision_stop_distance = float(
            self.get_parameter("collision_stop_distance").value
        )
        self.collision_distance = (
            2.0 * float(self.get_parameter("car_radius").value)
            if collision_stop_distance < 0.0
            else collision_stop_distance
        )
        self.max_planar_acceleration = float(
            self.get_parameter("max_planar_acceleration").value
        )
        if self.max_planar_acceleration < 0.0:
            raise ValueError("max_planar_acceleration must be nonnegative.")
        self.dynamic_reference_generator = bool(
            self.get_parameter("dynamic_reference_generator").value
        )
        self.max_planar_speed = float(self.get_parameter("max_planar_speed").value)
        self.max_planar_jerk = float(self.get_parameter("max_planar_jerk").value)
        if self.max_planar_speed < 0.0:
            raise ValueError("max_planar_speed must be nonnegative.")
        if self.max_planar_jerk < 0.0:
            raise ValueError("max_planar_jerk must be nonnegative.")
        if self.dynamic_reference_generator and (
            self.max_planar_acceleration <= 0.0
            or self.max_planar_speed <= 0.0
            or self.max_planar_jerk <= 0.0
        ):
            raise ValueError(
                "dynamic_reference_generator requires positive "
                "max_planar_acceleration, max_planar_speed, and max_planar_jerk."
            )
        self.sim_command_delay = float(
            self.get_parameter("sim_command_delay").value
        )
        self.sim_command_delay_jitter = float(
            self.get_parameter("sim_command_delay_jitter").value
        )
        self.sim_position_noise_std = float(
            self.get_parameter("sim_position_noise_std").value
        )
        if min(
            self.sim_command_delay,
            self.sim_command_delay_jitter,
            self.sim_position_noise_std,
        ) < 0.0:
            raise ValueError("Simulation stress-test parameters must be nonnegative.")
        if self.sim_command_delay_jitter > self.sim_command_delay:
            raise ValueError(
                "sim_command_delay_jitter cannot exceed sim_command_delay."
            )
        if self.mode != "sim" and any(
            value > 0.0
            for value in (
                self.sim_command_delay,
                self.sim_command_delay_jitter,
                self.sim_position_noise_std,
            )
        ):
            raise ValueError("Simulation stress-test parameters require mode='sim'.")
        self.sim_rng = np.random.default_rng(
            int(self.get_parameter("sim_noise_seed").value)
        )
        state_lower, state_upper = self.env.state_lim()
        self.velocity_lower = np.asarray(state_lower[2:4], dtype=float)
        self.velocity_upper = np.asarray(state_upper[2:4], dtype=float)

        robot_items = load_crazyflies(Path(self.get_parameter("crazyflies_yaml").value))
        self.robot_names = [name for name, _ in robot_items]
        if len(self.robot_names) != N_AGENTS:
            raise RuntimeError(f"Expected {N_AGENTS} enabled Crazyflies, found {len(self.robot_names)}")
        self.initial_positions = {
            name: np.asarray(position, dtype=float)
            for name, position in robot_items
        }
        starts_xy = np.asarray([position[:2] for _, position in robot_items], dtype=float)
        goal_assignment = str(self.get_parameter("goal_assignment").value).lower()
        if goal_assignment == "opposite":
            self.goals_xy = np.roll(starts_xy, shift=N_AGENTS // 2, axis=0)
        elif goal_assignment == "clockwise_rings":
            clockwise_targets = {
                "cf1": "cf2", "cf2": "cf3", "cf3": "cf4", "cf4": "cf1",
                "cf5": "cf6", "cf6": "cf7", "cf7": "cf8", "cf8": "cf5",
            }
            self.goals_xy = np.asarray(
                [self.initial_positions[clockwise_targets[name]][:2]
                 for name in self.robot_names],
                dtype=float,
            )
        else:
            raise ValueError(
                "goal_assignment must be 'opposite' or 'clockwise_rings'"
            )
        self.goals_jax = jnp.asarray(self.goals_xy, dtype=jnp.float32)

        if self.controller == "centralized_qp":
            relaxation_penalty = float(
                self.get_parameter("qp_relaxation_penalty").value
            )

            def infer(positions_xy, velocities_xy, goals_xy, obstacle_state):
                graph = make_graph(
                    self.env, obstacle_state, positions_xy, velocities_xy, goals_xy
                )
                action, relaxation = self.algo.get_qp_action(
                    graph, relax_penalty=relaxation_penalty
                )
                cbf = jnp.min(self.algo.get_cbf(graph), axis=1)
                return self.env.clip_action(action), cbf, relaxation
        else:
            def infer(positions_xy, velocities_xy, goals_xy, obstacle_state):
                graph = make_graph(
                    self.env, obstacle_state, positions_xy, velocities_xy, goals_xy
                )
                action = self.env.clip_action(self.algo.act(graph))
                return action, self.algo.get_cbf(graph), jnp.zeros((N_AGENTS, 3))

        self.infer_fn = jax.jit(infer)
        warmup = self.infer_fn(
            jnp.asarray(starts_xy, dtype=jnp.float32),
            jnp.zeros((N_AGENTS, 2), dtype=jnp.float32),
            self.goals_jax,
            self.obstacle_state,
        )
        jax.block_until_ready(warmup)

        self.command_publishers = {
            name: self.create_publisher(FullState, f"/{name}/cmd_full_state", 1)
            for name in self.robot_names
        }
        self.state_pub = self.create_publisher(Float64MultiArray, "/gcbf/state", 1)
        self.action_pub = self.create_publisher(Float64MultiArray, "/gcbf/action", 1)
        self.cbf_pub = self.create_publisher(Float64MultiArray, "/gcbf/cbf", 1)
        self.qp_relaxation_pub = self.create_publisher(
            Float64MultiArray, "/gcbf/qp_relaxation", 1
        )
        self.gcbf_acceleration_pub = self.create_publisher(
            Float64MultiArray, "/gcbf/gcbf_acceleration", 1
        )
        self.command_acceleration_pub = self.create_publisher(
            Float64MultiArray, "/gcbf/command_acceleration", 1
        )
        self.position_reference_pub = self.create_publisher(
            Float64MultiArray, "/gcbf/position_reference", 1
        )

        self.callback_group = ReentrantCallbackGroup()
        sensor_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        if self.mode == "sim":
            self.state_subscription = self.create_subscription(
                TFMessage, "/tf", self.tf_callback, sensor_qos,
                callback_group=self.callback_group,
            )
            self.arm_client = None
            self.arm_clients = {}
            self.notify_stop_clients = {}
            self.status_subscriptions = []
        else:
            self.state_subscription = self.create_subscription(
                NamedPoseArray, "/poses", self.poses_callback, sensor_qos,
                callback_group=self.callback_group,
            )
            self.arm_client = self.create_client(Arm, "/all/arm")
            self.arm_clients = {
                name: self.create_client(Arm, f"/{name}/arm")
                for name in self.robot_names
            }
            self.notify_stop_clients = {
                name: self.create_client(
                    NotifySetpointsStop, f"/{name}/notify_setpoints_stop"
                )
                for name in self.robot_names
            }
            self.status_subscriptions = [
                self.create_subscription(
                    Status,
                    f"/{name}/status",
                    lambda msg, robot_name=name: self.status_callback(robot_name, msg),
                    10,
                    callback_group=self.callback_group,
                )
                for name in self.robot_names
            ]

        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.pending_snapshot = None
        self.inference_running = False
        self.sequence = 0
        self.previous_positions = None
        self.previous_source_time = None
        self.position_history = []
        self.velocity_filter_window = float(
            self.get_parameter("velocity_filter_window").value
        )
        self.velocity_filter_min_samples = int(
            self.get_parameter("velocity_filter_min_samples").value
        )
        if self.velocity_filter_window < 0.0:
            raise ValueError("velocity_filter_window must be nonnegative")
        if self.velocity_filter_min_samples < 2:
            raise ValueError("velocity_filter_min_samples must be at least 2")
        self.last_snapshot_receive_time = None
        self.incomplete_pose_start_time = None
        self.pose_loss_start_time = None
        self.last_rejected_pose_reason = None
        self.last_rejected_pose_log_time = -np.inf
        self.last_policy_source_time = None
        self.dynamic_position_reference = None
        self.dynamic_velocity_reference = None
        self.dynamic_acceleration_reference = None
        self.delayed_reference_heap = []
        self.delayed_reference_sequence = 0
        self.last_delivered_reference_sequence = -1
        self.last_min_distance_log_time = -np.inf
        self.last_takeoff_log_time = -np.inf
        self.last_policy_command_log_time = -np.inf

        self.phase = "waiting"
        self.status_by_name = {}
        self.preflight_logged = False
        self.preflight_ready_logged = False
        self.last_wait_log_time = -np.inf
        self.phase_start_time = None
        self.takeoff_start_positions = None
        self.policy_start_time = None
        self.hold_positions = None
        self.landing_start_positions = None
        self.landing_active_names = set()
        self.landing_pending_disarm = {}
        self.landing_settling = {}
        self.arm_requested = False
        self.disarm_requested = False

        self.print_latency = bool(self.get_parameter("print_latency").value)
        self.latency_samples = {}
        self.latency_counts = {
            "processed": 0,
            "replaced": 0,
            "stale": 0,
            "incomplete": 0,
            "deadline_miss": 0,
        }
        self.last_latency_report = time.perf_counter()
        if self.print_latency:
            self.get_logger().info(
                f"JAX backend={jax.default_backend()} devices={jax.devices()}"
            )
        self.get_logger().info(
            f"Single-node {self.mode} controller listening on "
            f"{'/tf' if self.mode == 'sim' else '/poses'}; "
            f"timer-driven control at {self.rate_hz:.1f} Hz."
        )
        self.get_logger().info(
            f"{self.controller} environment mass={self.mass:.6f} kg; "
            f"physical car_radius={0.5 * self.collision_distance:.3f} m; "
            f"planning car_radius={float(self.env._params['car_radius']):.3f} m."
        )
        if self.velocity_filter_window > 0.0:
            self.get_logger().info(
                "Timestamp-aware velocity filtering enabled: "
                f"causal linear fit over {self.velocity_filter_window * 1000.0:.1f} ms, "
                f"minimum {self.velocity_filter_min_samples} samples."
            )
        if self.max_planar_acceleration > 0.0:
            self.get_logger().info(
                "Deployment acceleration norm clamp enabled: "
                f"{self.max_planar_acceleration:.3f} m/s^2."
            )
        else:
            self.get_logger().info("Deployment acceleration norm clamp disabled.")
        if self.dynamic_reference_generator:
            self.get_logger().info(
                "Dynamic planar reference generator enabled: "
                f"acceleration={self.max_planar_acceleration:.3f} m/s^2, "
                f"speed={self.max_planar_speed:.3f} m/s, "
                f"jerk={self.max_planar_jerk:.3f} m/s^3."
            )
        if self.mode == "sim" and (
            self.sim_command_delay > 0.0 or self.sim_position_noise_std > 0.0
        ):
            self.get_logger().info(
                "Simulation deployment stress enabled: "
                f"command delay={self.sim_command_delay * 1000.0:.1f} ms, "
                f"delay jitter=±{self.sim_command_delay_jitter * 1000.0:.1f} ms, "
                f"position noise std={self.sim_position_noise_std * 1000.0:.1f} mm."
            )
        self.control_timer = self.create_timer(
            self.min_period,
            self.control_timer_callback,
            callback_group=self.callback_group,
        )
        self.wait_timer = self.create_timer(2.0, self.log_waiting_inputs)

    def now_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def stamp_seconds(stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def status_callback(self, name, msg):
        self.status_by_name[name] = {
            "timestamp": self.stamp_seconds(msg.header.stamp),
            "supervisor": int(msg.supervisor_info),
            "battery": float(msg.battery_voltage),
            "pm_state": int(msg.pm_state),
            "rssi": int(msg.rssi),
            "num_rx_unicast": int(msg.num_rx_unicast),
            "num_tx_unicast": int(msg.num_tx_unicast),
        }

    @staticmethod
    def supervisor_flags(supervisor):
        flags = [
            ("can_arm", 1),
            ("is_armed", 2),
            ("auto_arm", 4),
            ("can_fly", 8),
            ("is_flying", 16),
            ("is_tumbled", 32),
            ("is_locked", 64),
        ]
        return [name for name, bit in flags if supervisor & bit]

    def log_preflight_positions(self, snapshot):
        tolerance = float(self.get_parameter("initial_position_tolerance").value)
        self.get_logger().info("Preflight initial position check against crazyflies YAML:")
        failures = []
        for name, measured in zip(self.robot_names, snapshot.state[:, 0:3]):
            expected = self.initial_positions[name]
            error = float(np.linalg.norm(measured - expected))
            message = (
                f"{name}: mocap=[{measured[0]:+.3f}, {measured[1]:+.3f}, {measured[2]:+.3f}] "
                f"yaml=[{expected[0]:+.3f}, {expected[1]:+.3f}, {expected[2]:+.3f}] "
                f"error={error:.3f} m"
            )
            if error > tolerance:
                self.get_logger().warn(yellow_text(message))
                failures.append(name)
            else:
                self.get_logger().info(message)
        return failures

    def log_preflight_statuses(self):
        min_rssi = int(self.get_parameter("min_radio_rssi").value)
        failures = []
        missing = [name for name in self.robot_names if name not in self.status_by_name]
        for name in self.robot_names:
            status = self.status_by_name.get(name)
            if status is None:
                continue
            supervisor = status["supervisor"]
            flags = self.supervisor_flags(supervisor)
            self.get_logger().info(
                f"/{name} status preflight radio: supervisor={supervisor} "
                f"flags={flags} battery={status['battery']:.2f}V "
                f"pm_state={status['pm_state']} rssi={status['rssi']} "
                f"rx_unicast={status['num_rx_unicast']} "
                f"tx_unicast={status['num_tx_unicast']}"
            )
            if supervisor & 64:
                failures.append(f"{name}: locked")
            if not supervisor & 8 or status["battery"] <= 0.0:
                failures.append(
                    f"{name}: not ready to fly "
                    f"(supervisor={supervisor}, battery={status['battery']:.2f}V)"
                )
            if status["rssi"] < min_rssi:
                failures.append(
                    f"{name}: RSSI {status['rssi']} below minimum {min_rssi}"
                )
        failures.extend(f"{name}: no status sample" for name in missing)
        return failures

    def preflight_passed(self, snapshot):
        if self.mode != "real":
            return True

        now = self.now_seconds()
        if len(self.status_by_name) < len(self.robot_names):
            if now - self.last_wait_log_time >= 1.0:
                missing = [
                    name for name in self.robot_names
                    if name not in self.status_by_name
                ]
                self.get_logger().info(
                    "Waiting for status samples before GCBF launch: "
                    + ", ".join(missing)
                )
                self.last_wait_log_time = now
            return False

        if not self.preflight_logged:
            position_failures = self.log_preflight_positions(snapshot)
            status_failures = self.log_preflight_statuses()
            self.preflight_logged = True
            if position_failures or status_failures:
                failures = [
                    *(f"{name}: initial position error too large" for name in position_failures),
                    *status_failures,
                ]
                raise RuntimeError("GCBF preflight failed: " + "; ".join(failures))

        if not self.preflight_ready_logged:
            self.get_logger().info("Preflight passed; starting streamed GCBF flight.")
            self.preflight_ready_logged = True
        return True

    def tf_callback(self, msg):
        positions = {}
        stamps = []
        for transform in msg.transforms:
            child = transform.child_frame_id.lstrip("/")
            parent = transform.header.frame_id.lstrip("/")
            if parent != "world" or child not in self.robot_names:
                continue
            p = transform.transform.translation
            positions[child] = np.array([p.x, p.y, p.z], dtype=float)
            stamps.append(self.stamp_seconds(transform.header.stamp))
        if len(positions) != N_AGENTS:
            self.latency_counts["incomplete"] += 1
            return
        self.accept_positions(positions, max(stamps) if stamps else 0.0)

    def poses_callback(self, msg):
        positions = {
            item.name: np.array(
                [item.pose.position.x, item.pose.position.y, item.pose.position.z], dtype=float
            )
            for item in msg.poses
            if item.name in self.robot_names
        }
        frame_id = msg.header.frame_id.lstrip("/")
        if len(positions) != N_AGENTS or frame_id != "world":
            self.latency_counts["incomplete"] += 1
            if self.mode == "real" and self.phase in {
                "takeoff", "policy", "hold", "landing"
            }:
                now = self.now_seconds()
                if self.incomplete_pose_start_time is None:
                    self.incomplete_pose_start_time = now
            missing = [name for name in self.robot_names if name not in positions]
            reasons = []
            if frame_id != "world":
                reasons.append(f"frame_id={frame_id!r}, expected 'world'")
            if missing:
                reasons.append("missing=" + ",".join(missing))
            self.last_rejected_pose_reason = "; ".join(reasons)
            now = self.now_seconds()
            if self.phase == "policy" and now - self.last_rejected_pose_log_time >= 1.0:
                self.get_logger().warn(
                    yellow_text(
                        f"Rejected /poses during policy: {self.last_rejected_pose_reason} "
                        f"(present={len(positions)}/{N_AGENTS})"
                    )
                )
                self.last_rejected_pose_log_time = now
            return
        self.incomplete_pose_start_time = None
        self.last_rejected_pose_reason = None
        self.accept_positions(positions, self.stamp_seconds(msg.header.stamp))

    def accept_positions(self, positions, source_time):
        assembly_start = time.perf_counter()
        received_time = self.now_seconds()
        self.last_snapshot_receive_time = received_time
        if source_time <= 0.0:
            source_time = received_time
        ordered_positions = np.asarray([positions[name] for name in self.robot_names], dtype=float)
        velocity_start = time.perf_counter()
        with self.state_lock:
            if self.previous_source_time is not None and source_time <= self.previous_source_time:
                return
            if self.previous_positions is None:
                velocities = np.zeros_like(ordered_positions)
            else:
                dt = source_time - self.previous_source_time
                velocities = (ordered_positions - self.previous_positions) / dt
            if self.velocity_filter_window > 0.0:
                self.position_history.append(
                    (source_time, ordered_positions.copy())
                )
                cutoff = source_time - self.velocity_filter_window
                self.position_history = [
                    item for item in self.position_history
                    if item[0] >= cutoff
                ]
                if len(self.position_history) >= self.velocity_filter_min_samples:
                    history_times = np.asarray(
                        [item[0] for item in self.position_history],
                        dtype=float,
                    )
                    history_positions = np.stack(
                        [item[1] for item in self.position_history],
                        axis=0,
                    )
                    centered_times = history_times - np.mean(history_times)
                    denominator = float(np.dot(centered_times, centered_times))
                    if denominator > 0.0:
                        velocities = np.tensordot(
                            centered_times,
                            history_positions,
                            axes=(0, 0),
                        ) / denominator
            self.previous_positions = ordered_positions.copy()
            self.previous_source_time = source_time

            rows = np.zeros((N_AGENTS, STATE_WIDTH), dtype=float)
            rows[:, 0:3] = ordered_positions
            rows[:, 3:6] = velocities
            rows[:, 6:8] = self.goals_xy
            rows[:, 8] = self.rate_hz
            self.sequence += 1
            snapshot = StateSnapshot(self.sequence, source_time, received_time, rows)
        self.record_latency("velocity", time.perf_counter() - velocity_start)
        self.record_latency("state_assembly", time.perf_counter() - assembly_start)
        self.queue_snapshot(snapshot)

    def log_waiting_inputs(self):
        if self.phase != "waiting":
            return
        if self.mode == "sim":
            source = "/tf"
        else:
            source = "/poses"
        if self.last_snapshot_receive_time is None:
            status_count = len(self.status_by_name)
            message = (
                f"Waiting for complete {source} snapshot before preflight "
                f"(status samples {status_count}/{len(self.robot_names)})."
            )
            if self.last_rejected_pose_reason:
                message += f" Last rejected pose message: {self.last_rejected_pose_reason}."
            self.get_logger().info(message)

    def queue_snapshot(self, snapshot):
        with self.lock:
            if self.pending_snapshot is not None:
                self.latency_counts["replaced"] += 1
            self.pending_snapshot = snapshot

    def control_timer_callback(self):
        if self.check_pose_loss_fail_safe():
            return
        with self.lock:
            if self.inference_running or self.pending_snapshot is None:
                return
            snapshot = self._claim_pending_locked()
        self.process_snapshot(snapshot)

    def check_pose_loss_fail_safe(self):
        if self.mode != "real":
            return False
        now = self.now_seconds()
        if self.phase == "pose_loss":
            self.run_pose_loss_fail_safe(now)
            return True
        loss_start_time = self.incomplete_pose_start_time
        if loss_start_time is None:
            loss_start_time = self.last_snapshot_receive_time
        if (
            loss_start_time is None
            or self.phase not in {"takeoff", "policy", "hold", "landing"}
            or now - loss_start_time
            < float(self.get_parameter("pose_loss_timeout").value)
        ):
            return False
        with self.state_lock:
            if self.previous_positions is None:
                return False
            self.hold_positions = self.previous_positions.copy()
        self.pose_loss_start_time = now
        self.phase = "pose_loss"
        self.get_logger().error(
            red_text(
                "Sustained incomplete Vicon tracking; starting timer-driven "
                "fleet hold, landing, and disarm."
            )
        )
        self.run_pose_loss_fail_safe(now)
        return True

    def run_pose_loss_fail_safe(self, now):
        zeros = np.zeros((N_AGENTS, 3), dtype=float)
        elapsed = now - self.pose_loss_start_time
        if elapsed < POSE_LOSS_HOLD_SECONDS:
            self.dispatch_reference(self.hold_positions, zeros, zeros, now)
            return
        landing_elapsed = elapsed - POSE_LOSS_HOLD_SECONDS
        refs = self.vertical_reference(
            self.hold_positions,
            LAND_HEIGHT,
            LAND_DURATION,
            landing_elapsed,
        )
        self.dispatch_reference(*refs, now)
        if landing_elapsed < LAND_DURATION or self.disarm_requested:
            return
        if self.arm_client.service_is_ready():
            request = Arm.Request()
            request.arm = False
            self.arm_client.call_async(request)
            self.disarm_requested = True
            self.phase = "landed"
            self.get_logger().error(
                red_text("Pose-loss landing elapsed; requested hardware disarm.")
            )

    def _claim_pending_locked(self):
        snapshot = self.pending_snapshot
        self.pending_snapshot = None
        self.inference_running = True
        return snapshot

    def process_snapshot(self, snapshot):
        cycle_start = time.perf_counter()
        try:
            state_age = self.now_seconds() - snapshot.source_time
            self.record_latency("state_age_at_start", state_age)
            self.record_latency("callback_and_rate_wait", self.now_seconds() - snapshot.received_time)
            if state_age > float(self.get_parameter("pose_timeout").value):
                self.latency_counts["stale"] += 1
                return

            self.publish_array(self.state_pub, snapshot.state, "state_width", STATE_WIDTH)
            if self.phase != "policy":
                zero_acceleration = np.zeros((N_AGENTS, 2), dtype=float)
                self.publish_array(
                    self.gcbf_acceleration_pub,
                    zero_acceleration,
                    "gcbf_acceleration_width",
                    2,
                )
                self.publish_array(
                    self.qp_relaxation_pub,
                    np.zeros((N_AGENTS, 3), dtype=float),
                    "qp_relaxation_width",
                    3,
                )
            if self.phase == "waiting":
                if not self.preflight_passed(snapshot):
                    return
                self.start_takeoff(snapshot)

            if self.phase == "takeoff":
                self.run_takeoff(snapshot)
            elif self.phase == "policy":
                self.run_policy(snapshot)
            elif self.phase == "hold":
                self.run_hold(snapshot)
            elif self.phase == "landing":
                self.run_landing(snapshot)
            elif self.phase == "hold_fault":
                zeros = np.zeros((N_AGENTS, 3), dtype=float)
                self.publish_reference(self.hold_positions, zeros, zeros, snapshot)
            elif self.phase == "landed":
                self.run_landed(snapshot)

            self.latency_counts["processed"] += 1
        finally:
            cycle_time = time.perf_counter() - cycle_start
            self.record_latency("cycle", cycle_time)
            if cycle_time > self.min_period:
                self.latency_counts["deadline_miss"] += 1
            self.finish_cycle()
            self.maybe_report_latency(snapshot.sequence)

    def finish_cycle(self):
        with self.lock:
            self.inference_running = False

    def start_takeoff(self, snapshot):
        if self.mode == "real" and not self.arm_requested:
            if not self.arm_client.service_is_ready():
                return
            request = Arm.Request()
            request.arm = True
            self.arm_client.call_async(request)
            self.arm_requested = True
        self.takeoff_start_positions = snapshot.state[:, 0:3].copy()
        self.phase_start_time = snapshot.source_time
        self.phase = "takeoff"
        self.get_logger().info("Starting low-level cmd_full_state takeoff stream.")

    @staticmethod
    def smooth_step(tau):
        tau = float(np.clip(tau, 0.0, 1.0))
        value = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
        first = 30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4
        second = 60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3
        return value, first, second

    def vertical_reference(self, start_positions, target_z, duration, elapsed):
        duration = max(float(duration), 1e-6)
        value, first, second = self.smooth_step(elapsed / duration)
        delta = target_z - start_positions[:, 2]
        positions = start_positions.copy()
        positions[:, 2] = start_positions[:, 2] + delta * value
        velocities = np.zeros((N_AGENTS, 3), dtype=float)
        velocities[:, 2] = delta * first / duration
        accelerations = np.zeros((N_AGENTS, 3), dtype=float)
        accelerations[:, 2] = delta * second / duration**2
        return positions, velocities, accelerations

    def run_takeoff(self, snapshot):
        now = snapshot.source_time
        duration = float(self.get_parameter("takeoff_duration").value)
        elapsed = now - self.phase_start_time
        refs = self.vertical_reference(
            self.takeoff_start_positions,
            float(self.get_parameter("hover_height").value),
            duration,
            elapsed,
        )
        self.publish_reference(*refs, snapshot)
        if elapsed < duration:
            return
        hover_height = float(self.get_parameter("hover_height").value)
        hover_epsilon = float(self.get_parameter("hover_epsilon").value)
        z_errors = np.abs(snapshot.state[:, 2] - hover_height)
        if np.all(z_errors <= hover_epsilon) or self.mode == "sim":
            if self.mode == "sim" and np.any(z_errors > hover_epsilon):
                self.get_logger().warn(
                    yellow_text(
                        "Sim takeoff duration elapsed before all z estimates reached "
                        f"hover band: z_min={np.min(snapshot.state[:, 2]):.3f}, "
                        f"z_max={np.max(snapshot.state[:, 2]):.3f}, "
                        f"target={hover_height:.3f}; starting GCBF policy."
                    )
                )
            self.phase = "policy"
            self.policy_start_time = now
            self.last_policy_source_time = None
            self.dynamic_position_reference = None
            self.dynamic_velocity_reference = None
            self.dynamic_acceleration_reference = None
            self.get_logger().info("Hover confirmed; starting GCBF policy.")
        elif elapsed > float(self.get_parameter("takeoff_timeout").value):
            self.get_logger().error(red_text("Timed out waiting for low-level takeoff confirmation."))
            self.hold_positions = snapshot.state[:, 0:3].copy()
            self.phase = "hold_fault"
        elif now - self.last_takeoff_log_time >= 1.0:
            self.get_logger().info(
                f"Takeoff waiting for hover band: z_min={np.min(snapshot.state[:, 2]):.3f}, "
                f"z_max={np.max(snapshot.state[:, 2]):.3f}, target={hover_height:.3f}, "
                f"epsilon={hover_epsilon:.3f}"
            )
            self.last_takeoff_log_time = now

    def run_policy(self, snapshot):
        policy_test_duration = float(
            self.get_parameter("policy_test_duration").value
        )
        if (
            policy_test_duration > 0.0
            and snapshot.source_time - self.policy_start_time >= policy_test_duration
        ):
            self.get_logger().info(
                f"Policy timing test reached {policy_test_duration:.2f} s; "
                "starting automatic hold and landing."
            )
            self.hold_positions = snapshot.state[:, 0:3].copy()
            self.phase_start_time = snapshot.source_time
            self.phase = "hold"
            return

        if snapshot.source_time - self.policy_start_time > float(self.get_parameter("max_runtime").value):
            self.get_logger().info("GCBF actor reached max runtime; holding position.")
            self.hold_positions = snapshot.state[:, 0:3].copy()
            self.phase = "hold_fault"
            return

        state = snapshot.state
        positions_xy = state[:, 0:2]
        measured_velocities_xy = state[:, 3:5]
        velocities_xy = np.clip(
            measured_velocities_xy, self.velocity_lower, self.velocity_upper
        )
        goals_xy = state[:, 6:8]

        section_start = time.perf_counter()
        policy_positions_xy = positions_xy
        if self.mode == "sim" and self.sim_position_noise_std > 0.0:
            policy_positions_xy = positions_xy + self.sim_rng.normal(
                0.0, self.sim_position_noise_std, size=positions_xy.shape
            )
        positions_jax = jnp.asarray(policy_positions_xy, dtype=jnp.float32)
        velocities_jax = jnp.asarray(velocities_xy, dtype=jnp.float32)
        jax.block_until_ready((positions_jax, velocities_jax))
        self.record_latency("input_conversion", time.perf_counter() - section_start)

        section_start = time.perf_counter()
        action_jax, cbf_jax, relaxation_jax = self.infer_fn(
            positions_jax, velocities_jax, self.goals_jax, self.obstacle_state
        )
        jax.block_until_ready((action_jax, cbf_jax, relaxation_jax))
        action = np.asarray(action_jax, dtype=float)
        cbf = np.asarray(cbf_jax, dtype=float).reshape(N_AGENTS)
        relaxation = np.asarray(relaxation_jax, dtype=float).reshape(N_AGENTS, 3)
        self.record_latency("compiled_inference", time.perf_counter() - section_start)

        action_age = self.now_seconds() - snapshot.source_time
        self.record_latency("action_age", action_age)
        if action_age > float(self.get_parameter("max_action_age").value):
            self.latency_counts["stale"] += 1
            return

        section_start = time.perf_counter()
        min_distance = pairwise_min_distance(positions_xy)
        if self.collision_distance > 0.0 and min_distance < self.collision_distance:
            self.get_logger().error(
                red_text(
                    f"Pairwise distance {min_distance:.3f} below collision boundary "
                    f"{self.collision_distance:.3f}; holding position."
                )
            )
            self.hold_positions = state[:, 0:3].copy()
            self.phase = "hold_fault"
            return

        self.publish_array(self.action_pub, action, "action_width", 2)
        self.publish_array(self.cbf_pub, cbf.reshape(N_AGENTS, 1), "cbf_width", 1)
        self.publish_array(
            self.qp_relaxation_pub,
            relaxation,
            "qp_relaxation_width",
            3,
        )
        goal_errors = np.linalg.norm(goals_xy - positions_xy, axis=1)
        if np.all(goal_errors < float(self.get_parameter("goal_tolerance").value)):
            self.get_logger().info("All Crazyflies reached their goals; starting hold.")
            self.hold_positions = state[:, 0:3].copy()
            self.phase_start_time = snapshot.source_time
            self.phase = "hold"
            return

        gcbf_accel_xy = action / self.mass
        if self.dynamic_reference_generator:
            positions, velocities, accelerations = self.generate_dynamic_reference(
                snapshot, state, measured_velocities_xy, gcbf_accel_xy
            )
        else:
            accel_xy = self.clamp_planar_acceleration(gcbf_accel_xy)
            propagation_dt = float(self.get_parameter("lookahead_dt").value)
            self.last_policy_source_time = snapshot.source_time
            velocities = np.zeros((N_AGENTS, 3), dtype=float)
            velocities[:, 0:2] = np.clip(
                velocities_xy + accel_xy * propagation_dt,
                self.velocity_lower,
                self.velocity_upper,
            )
            position_ref_accel_scale = float(
                self.get_parameter("position_ref_accel_scale").value
            )
            positions = state[:, 0:3].copy()
            positions[:, 0:2] = (
                positions_xy
                + velocities_xy * propagation_dt
                + position_ref_accel_scale * 0.5 * accel_xy * propagation_dt**2
            )
            positions[:, 2] = float(self.get_parameter("hover_height").value)
            accelerations = np.zeros((N_AGENTS, 3), dtype=float)
            # The policy output is force and gcbf_accel_xy is its acceleration
            # interpretation. Send it directly; reconstructing acceleration
            # from a clipped velocity reference can amplify the command.
            accelerations[:, 0:2] = accel_xy
        self.publish_array(
            self.gcbf_acceleration_pub,
            gcbf_accel_xy,
            "gcbf_acceleration_width",
            2,
        )
        if bool(self.get_parameter("policy_diagnostics").value) and self.mode == "real":
            self.log_initial_policy_commands(
                snapshot,
                positions_xy,
                goals_xy,
                action,
                positions,
                velocities,
                accelerations,
            )
        if bool(self.get_parameter("policy_diagnostics").value):
            self.log_policy_safety_and_tracking(
                snapshot,
                positions_xy,
                measured_velocities_xy,
                velocities[:, 0:2],
                min_distance,
            )
        self.record_latency("safety_lookahead", time.perf_counter() - section_start)
        self.publish_reference(positions, velocities, accelerations, snapshot)

    @staticmethod
    def format_vec(values):
        return "[" + ", ".join(f"{float(value):+.3f}" for value in values) + "]"

    def clamp_planar_acceleration(self, values):
        result = np.asarray(values, dtype=float).copy()
        if self.max_planar_acceleration <= 0.0:
            return result
        norms = np.linalg.norm(result, axis=1)
        scales = np.minimum(
            1.0,
            self.max_planar_acceleration / np.maximum(norms, 1e-12),
        )
        return result * scales[:, None]

    @staticmethod
    def clamp_planar_norm(values, maximum):
        result = np.asarray(values, dtype=float).copy()
        if maximum <= 0.0:
            return result
        norms = np.linalg.norm(result, axis=1)
        scales = np.minimum(1.0, maximum / np.maximum(norms, 1e-12))
        return result * scales[:, None]

    def generate_dynamic_reference(
        self, snapshot, state, measured_velocities_xy, desired_acceleration_xy
    ):
        if self.dynamic_position_reference is None:
            self.dynamic_position_reference = state[:, 0:3].copy()
            self.dynamic_position_reference[:, 2] = float(
                self.get_parameter("hover_height").value
            )
            self.dynamic_velocity_reference = np.zeros((N_AGENTS, 3), dtype=float)
            self.dynamic_velocity_reference[:, 0:2] = self.clamp_planar_norm(
                measured_velocities_xy, self.max_planar_speed
            )
            self.dynamic_acceleration_reference = np.zeros(
                (N_AGENTS, 3), dtype=float
            )
            dt = self.min_period
        else:
            dt = snapshot.source_time - self.last_policy_source_time
            dt = float(np.clip(dt, 1e-4, 2.0 * self.min_period))

        desired_acceleration_xy = self.clamp_planar_acceleration(
            desired_acceleration_xy
        )
        acceleration_delta = (
            desired_acceleration_xy
            - self.dynamic_acceleration_reference[:, 0:2]
        )
        acceleration_delta = self.clamp_planar_norm(
            acceleration_delta, self.max_planar_jerk * dt
        )
        acceleration_xy = self.dynamic_acceleration_reference[:, 0:2] + acceleration_delta
        acceleration_xy = self.clamp_planar_acceleration(acceleration_xy)

        previous_velocity_xy = self.dynamic_velocity_reference[:, 0:2].copy()
        velocity_xy = self.clamp_planar_norm(
            previous_velocity_xy + acceleration_xy * dt,
            self.max_planar_speed,
        )
        effective_acceleration_xy = (velocity_xy - previous_velocity_xy) / dt
        position_xy = (
            self.dynamic_position_reference[:, 0:2]
            + previous_velocity_xy * dt
            + 0.5 * effective_acceleration_xy * dt**2
        )

        self.dynamic_position_reference[:, 0:2] = position_xy
        self.dynamic_velocity_reference[:, 0:2] = velocity_xy
        self.dynamic_acceleration_reference[:, 0:2] = effective_acceleration_xy
        self.last_policy_source_time = snapshot.source_time
        return (
            self.dynamic_position_reference.copy(),
            self.dynamic_velocity_reference.copy(),
            self.dynamic_acceleration_reference.copy(),
        )

    def closest_pair_metrics(self, positions_xy, measured_velocities_xy):
        best = None
        for i in range(N_AGENTS):
            for j in range(i + 1, N_AGENTS):
                delta = positions_xy[j] - positions_xy[i]
                distance = float(np.linalg.norm(delta))
                if distance <= 1e-9:
                    direction = np.zeros(2, dtype=float)
                else:
                    direction = delta / distance
                relative_velocity = measured_velocities_xy[j] - measured_velocities_xy[i]
                closing_speed = -float(np.dot(relative_velocity, direction))
                relative_speed = float(np.linalg.norm(relative_velocity))
                if best is None or distance < best["distance"]:
                    best = {
                        "i": i,
                        "j": j,
                        "distance": distance,
                        "relative_speed": relative_speed,
                        "closing_speed": closing_speed,
                    }
        return best

    def log_policy_safety_and_tracking(
        self,
        snapshot,
        positions_xy,
        measured_velocities_xy,
        velocity_refs_xy,
        min_distance,
    ):
        if snapshot.source_time - self.last_min_distance_log_time < 1.0:
            return

        closest = self.closest_pair_metrics(positions_xy, measured_velocities_xy)
        velocity_errors = velocity_refs_xy - measured_velocities_xy
        velocity_error_norms = np.linalg.norm(velocity_errors, axis=1)
        worst_index = int(np.argmax(velocity_error_norms))
        parts = [
            f"Minimum pairwise distance: {min_distance:.3f} m",
            f"collision boundary: {self.collision_distance:.3f} m",
        ]
        if closest is not None:
            parts.append(
                "closest_pair="
                f"{self.robot_names[closest['i']]}-{self.robot_names[closest['j']]}"
            )
            parts.append(f"relative_speed={closest['relative_speed']:.3f} m/s")
            parts.append(f"closing_speed={closest['closing_speed']:.3f} m/s")
        parts.append(f"vel_error_mean={np.mean(velocity_error_norms):.3f} m/s")
        parts.append(
            f"vel_error_worst={self.robot_names[worst_index]}:"
            f"{velocity_error_norms[worst_index]:.3f} m/s"
        )
        self.get_logger().info("; ".join(parts))
        self.get_logger().info(
            f"{self.robot_names[worst_index]} velocity tracking: "
            f"measured={self.format_vec(measured_velocities_xy[worst_index])}, "
            f"ref={self.format_vec(velocity_refs_xy[worst_index])}, "
            f"error={self.format_vec(velocity_errors[worst_index])}"
        )
        self.last_min_distance_log_time = snapshot.source_time

    def log_initial_policy_commands(
        self,
        snapshot,
        positions_xy,
        goals_xy,
        action,
        position_refs,
        velocity_refs,
        acceleration_refs,
    ):
        if self.policy_start_time is None:
            return
        elapsed = snapshot.source_time - self.policy_start_time
        if elapsed < 0.0 or elapsed > 5.0:
            return
        if snapshot.source_time - self.last_policy_command_log_time < 1.0:
            return
        self.get_logger().info(
            f"Initial GCBF command references t={elapsed:.2f}s after handoff:"
        )
        for i, name in enumerate(self.robot_names):
            self.get_logger().info(
                f"{name}: pos_xy={self.format_vec(positions_xy[i])}, "
                f"goal_xy={self.format_vec(goals_xy[i])}, "
                f"action={self.format_vec(action[i])}, "
                f"pos_ref={self.format_vec(position_refs[i])}, "
                f"vel_ref={self.format_vec(velocity_refs[i])}, "
                f"acc_ref={self.format_vec(acceleration_refs[i])}"
            )
        self.last_policy_command_log_time = snapshot.source_time

    def run_hold(self, snapshot):
        zeros = np.zeros((N_AGENTS, 3), dtype=float)
        if snapshot.source_time - self.phase_start_time < GOAL_SETTLING_SECONDS:
            self.hold_positions[:, 0:2] = snapshot.state[:, 0:2]
        self.publish_reference(self.hold_positions, zeros, zeros, snapshot)
        if snapshot.source_time - self.phase_start_time >= GOAL_HOLD_SECONDS:
            # Land vertically from where each vehicle actually is when descent
            # begins. Reusing the earlier hold target makes a vehicle correct
            # residual hold error laterally while it is close to the floor.
            self.landing_start_positions = snapshot.state[:, 0:3].copy()
            self.landing_active_names = set(self.robot_names)
            self.landing_pending_disarm = {}
            self.landing_settling = {}
            self.phase_start_time = snapshot.source_time
            self.phase = "landing"
            self.get_logger().info("Starting low-level cmd_full_state landing stream.")

    def run_landing(self, snapshot):
        now = snapshot.source_time
        refs = self.vertical_reference(
            self.landing_start_positions,
            LAND_HEIGHT,
            LAND_DURATION,
            now - self.phase_start_time,
        )
        landing_disarm_height = float(
            self.get_parameter("landing_disarm_height").value
        )
        if self.mode == "real" and landing_disarm_height > 0.0:
            for name, deadline in list(self.landing_pending_disarm.items()):
                if now >= deadline and self.arm_clients[name].service_is_ready():
                    request = Arm.Request()
                    request.arm = False
                    self.arm_clients[name].call_async(request)
                    self.landing_settling[name] = now + 0.3
                    del self.landing_pending_disarm[name]
                    self.get_logger().info(f"{name}: requested hardware disarm.")
            for name, deadline in list(self.landing_settling.items()):
                if now >= deadline:
                    del self.landing_settling[name]

            for index, name in enumerate(self.robot_names):
                if (
                    name in self.landing_active_names
                    and snapshot.state[index, 2] <= landing_disarm_height
                    and self.notify_stop_clients[name].service_is_ready()
                    and self.arm_clients[name].service_is_ready()
                ):
                    request = NotifySetpointsStop.Request()
                    request.remain_valid_millisecs = 0
                    request.group_mask = 0
                    self.notify_stop_clients[name].call_async(request)
                    self.landing_active_names.remove(name)
                    self.landing_pending_disarm[name] = now + 0.1
                    self.get_logger().info(
                        f"{name}: reached touchdown height "
                        f"z={snapshot.state[index, 2]:.3f} m; stopping setpoints."
                    )

            self.dispatch_reference(
                refs[0], refs[1], refs[2], snapshot.source_time,
                active_names=self.landing_active_names,
            )
            if (
                not self.landing_active_names
                and not self.landing_pending_disarm
                and not self.landing_settling
            ):
                self.hold_positions = refs[0]
                self.phase = "landed"
                self.disarm_requested = True
                self.get_logger().info(
                    "Landing confirmed; all Crazyflies disarmed individually."
                )
            return

        self.publish_reference(*refs, snapshot)
        if now - self.phase_start_time < LAND_DURATION:
            return
        measured_z = snapshot.state[:, 2]
        measured_vz = snapshot.state[:, 5]
        tolerance = float(self.get_parameter("hover_epsilon").value)
        if not (np.all(np.abs(measured_z - LAND_HEIGHT) <= tolerance) and np.all(np.abs(measured_vz) <= tolerance)):
            return
        self.hold_positions = refs[0]
        self.phase = "landed"
        self.get_logger().info("Landing confirmed.")
        self.run_landed(snapshot)

    def run_landed(self, snapshot):
        if self.mode == "real" and not self.disarm_requested and self.arm_client.service_is_ready():
            request = Arm.Request()
            request.arm = False
            self.arm_client.call_async(request)
            self.disarm_requested = True
            self.get_logger().info("Landing confirmed; requested hardware disarm.")
        elif self.mode == "sim":
            zeros = np.zeros((N_AGENTS, 3), dtype=float)
            self.publish_reference(self.hold_positions, zeros, zeros, snapshot)

    def publish_reference(self, positions, velocities, accelerations, snapshot):
        if (
            self.mode == "sim"
            and self.phase == "policy"
            and self.sim_command_delay > 0.0
        ):
            jitter = self.sim_rng.uniform(
                -self.sim_command_delay_jitter,
                self.sim_command_delay_jitter,
            )
            delivery_time = (
                snapshot.source_time + self.sim_command_delay + float(jitter)
            )
            sequence = self.delayed_reference_sequence
            self.delayed_reference_sequence += 1
            heapq.heappush(
                self.delayed_reference_heap,
                (
                    delivery_time,
                    sequence,
                    np.asarray(positions, dtype=float).copy(),
                    np.asarray(velocities, dtype=float).copy(),
                    np.asarray(accelerations, dtype=float).copy(),
                    snapshot.source_time,
                ),
            )
            due = []
            while (
                self.delayed_reference_heap
                and self.delayed_reference_heap[0][0] <= snapshot.source_time
            ):
                due.append(heapq.heappop(self.delayed_reference_heap))
            if not due:
                return
            command = max(due, key=lambda item: item[1])
            if command[1] <= self.last_delivered_reference_sequence:
                return
            self.last_delivered_reference_sequence = command[1]
            self.dispatch_reference(
                command[2], command[3], command[4], command[5]
            )
            return

        if self.delayed_reference_heap:
            self.delayed_reference_heap.clear()
        self.dispatch_reference(
            positions, velocities, accelerations, snapshot.source_time
        )

    def dispatch_reference(
        self, positions, velocities, accelerations, source_time, active_names=None
    ):
        section_start = time.perf_counter()
        self.publish_array(
            self.command_acceleration_pub,
            np.asarray(accelerations)[:, 0:2],
            "command_acceleration_width",
            2,
        )
        self.publish_array(
            self.position_reference_pub,
            np.asarray(positions)[:, 0:2],
            "position_reference_width",
            2,
        )
        stamp = rclpy.time.Time(seconds=float(source_time)).to_msg()
        for name, position, velocity, acceleration in zip(
            self.robot_names, positions, velocities, accelerations
        ):
            if active_names is not None and name not in active_names:
                continue
            msg = FullState()
            msg.header.stamp = stamp
            msg.header.frame_id = "/world"
            msg.pose.position.x = float(position[0])
            msg.pose.position.y = float(position[1])
            msg.pose.position.z = float(position[2])
            msg.pose.orientation.w = 1.0
            msg.twist.linear.x = float(velocity[0])
            msg.twist.linear.y = float(velocity[1])
            msg.twist.linear.z = float(velocity[2])
            msg.acc.x = float(acceleration[0])
            msg.acc.y = float(acceleration[1])
            msg.acc.z = float(acceleration[2])
            self.command_publishers[name].publish(msg)
        self.record_latency("command_publish", time.perf_counter() - section_start)
        self.record_latency("state_to_command", self.now_seconds() - float(source_time))

    def publish_array(self, publisher, values, width_label, width):
        msg = Float64MultiArray()
        msg.layout.dim = [
            MultiArrayDimension(label="agents", size=N_AGENTS, stride=N_AGENTS * width),
            MultiArrayDimension(label=width_label, size=width, stride=width),
        ]
        msg.data = np.asarray(values, dtype=float).reshape(-1).tolist()
        publisher.publish(msg)

    def record_latency(self, name, seconds):
        if self.print_latency:
            self.latency_samples.setdefault(name, []).append(float(seconds) * 1e3)

    def maybe_report_latency(self, sequence):
        if not self.print_latency or time.perf_counter() - self.last_latency_report < 1.0:
            return
        parts = []
        for name, values in sorted(self.latency_samples.items()):
            if values:
                parts.append(
                    f"{name}=latest:{values[-1]:.2f}/mean:{np.mean(values):.2f}/max:{np.max(values):.2f}ms"
                )
        counts = ",".join(f"{key}:{value}" for key, value in self.latency_counts.items())
        self.get_logger().info(f"latency seq={sequence} {' '.join(parts)} counts={counts}")
        self.latency_samples.clear()
        self.last_latency_report = time.perf_counter()


def main(args=None):
    rclpy.init(args=args)
    node = GcbfActor()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
