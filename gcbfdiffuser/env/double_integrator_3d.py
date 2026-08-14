import functools as ft
import pathlib
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import (
    Action,
    AgentState,
    Array,
    Cost,
    Done,
    Info,
    Pos3d,
    Reward,
    State,
)
from ..utils.utils import jax_vmap, merge01
from .base import MultiAgentEnv, RolloutResult
from .obstacle import Obstacle, Sphere
from .plot import render_video
from .utils import get_lidar, get_node_goal_rng, inside_obstacles, lqr


class DoubleIntegrator3D(MultiAgentEnv):
    """
    Three-dimensional double-integrator environment for GCBF+.

    Agent state
    -----------
        x = [px, py, pz, vx, vy, vz]

    Control
    -------
        u = [ax, ay, az]

    Continuous-time dynamics
    ------------------------
        p_dot = v
        v_dot = u

    By default, the applied acceleration is projected onto

        ||u||_2 <= ma

    to reproduce the acceleration saturation used in the provided dynamics.

    Important
    ---------
    ``control_affine_dyn`` returns the unsaturated model

        x_dot = f(x) + g(x)u,

    because GCBF/CBF-QP differentiation requires a control-affine model.
    The action is projected onto the acceleration ball before simulation.
    If exact consistency with a box-constrained QP is more important, set
    ``use_norm_limit=False`` and rely only on ``action_lim``.
    """

    AGENT = 0
    GOAL = 1
    OBS = 2

    class EnvState(NamedTuple):
        agent: AgentState
        goal: State
        obstacle: Obstacle

        @property
        def n_agent(self) -> int:
            return self.agent.shape[0]

    EnvGraphsTuple = GraphsTuple[State, EnvState]

    PARAMS = {
        "drone_radius": 0.05,
        "comm_radius": 1.0,
        "n_rays": 16,
        "obs_len_range": [0.10, 0.60],
        "n_obs": 0,

        # Maximum Euclidean acceleration magnitude [m/s^2].
        "ma": 2.0,

        # Velocity clipping used by state_lim().
        "max_velocity": 1.5,

        # True reproduces the norm saturation in the supplied dynamics.
        "use_norm_limit": True,

        # Nominal LQR weights.
        "q_position": 5.0,
        "q_velocity": 1.0,
        "r_acceleration": 1.0,
    }

    PX, PY, PZ, VX, VY, VZ = range(6)
    AX, AY, AZ = range(3)

    def __init__(
        self,
        num_agents: int,
        area_size: float,
        max_step: int = 256,
        max_travel: float = None,
        dt: float = 0.03,
        params: dict = None,
    ):
        super().__init__(
            num_agents=num_agents,
            area_size=area_size,
            max_step=max_step,
            max_travel=max_travel,
            dt=dt,
            params=params,
        )

        self.create_obstacles = jax_vmap(Sphere.create)

        # The 3-D LiDAR utility returns a fixed number of hit points.
        self.n_rays = min(
            16,
            self._params["n_rays"] ** 2 // 2 + 2,
        )

        # Exact zero-order-hold discretization of the double integrator.
        eye3 = np.eye(3, dtype=np.float32)

        A_d = np.eye(self.state_dim, dtype=np.float32)
        A_d[:3, 3:] = self.dt * eye3

        B_d = np.zeros(
            (self.state_dim, self.action_dim),
            dtype=np.float32,
        )
        B_d[:3, :] = 0.5 * self.dt**2 * eye3
        B_d[3:, :] = self.dt * eye3

        q_pos = self._params["q_position"]
        q_vel = self._params["q_velocity"]
        r_acc = self._params["r_acceleration"]

        Q = np.diag(
            [q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]
        ).astype(np.float32)
        R = (r_acc * eye3).astype(np.float32)

        self._A = A_d
        self._B = B_d
        self._Q = Q
        self._R = R
        self._K = jnp.asarray(lqr(A_d, B_d, Q, R))

    # ------------------------------------------------------------------
    # Dimensions
    # ------------------------------------------------------------------

    @property
    def state_dim(self) -> int:
        return 6

    @property
    def node_dim(self) -> int:
        # One-hot node type: obstacle, goal, agent.
        return 3

    @property
    def edge_dim(self) -> int:
        # Relative [position(3), velocity(3)].
        return 6

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def comm_radius(self) -> float:
        return self._params["comm_radius"]

    @property
    def ma(self) -> float:
        return self._params["ma"]

    # ------------------------------------------------------------------
    # Reset and environment transition
    # ------------------------------------------------------------------

    def reset(self, key: Array) -> GraphsTuple:
        self._t = 0

        n_rng_obs = self._params["n_obs"]
        assert n_rng_obs >= 0

        # Random spherical obstacles.
        obstacle_key, key = jr.split(key)
        obs_pos = jr.uniform(
            obstacle_key,
            shape=(n_rng_obs, 3),
            minval=0.0,
            maxval=self.area_size,
        )

        radius_key, key = jr.split(key)
        obs_radius = jr.uniform(
            radius_key,
            shape=(n_rng_obs,),
            minval=self._params["obs_len_range"][0] / 2.0,
            maxval=self._params["obs_len_range"][1] / 2.0,
        )

        obstacles = self.create_obstacles(obs_pos, obs_radius)

        # Collision-free initial and goal positions.
        positions, goal_positions = get_node_goal_rng(
            key,
            self.area_size,
            3,
            obstacles,
            self.num_agents,
            4.0 * self._params["drone_radius"],
            self.max_travel,
        )

        zero_velocity = jnp.zeros((self.num_agents, 3))

        states = jnp.concatenate(
            [positions, zero_velocity],
            axis=1,
        )
        goals = jnp.concatenate(
            [goal_positions, zero_velocity],
            axis=1,
        )

        env_state = self.EnvState(
            agent=states,
            goal=goals,
            obstacle=obstacles,
        )
        return self.get_graph(env_state)

    def step(
        self,
        graph: EnvGraphsTuple,
        action: Action,
        get_eval_info: bool = False,
    ) -> Tuple[EnvGraphsTuple, Reward, Cost, Done, Info]:
        self._t += 1

        agent_states = graph.type_states(
            type_idx=self.AGENT,
            n_type=self.num_agents,
        )
        goal_states = graph.type_states(
            type_idx=self.GOAL,
            n_type=self.num_agents,
        )
        obstacles = graph.env_states.obstacle

        assert agent_states.shape == (
            self.num_agents,
            self.state_dim,
        )
        assert action.shape == (
            self.num_agents,
            self.action_dim,
        )

        applied_action = self.project_action(action)
        next_agent_states = self.agent_step_exact(
            agent_states,
            applied_action,
        )

        # The rollout code controls the episode horizon.
        done = jnp.array(False)

        nominal_action = self.u_ref(graph)
        reward = -jnp.mean(
            jnp.sum(
                (applied_action - nominal_action) ** 2,
                axis=1,
            )
        )
        cost = self.get_cost(graph)

        next_env_state = self.EnvState(
            agent=next_agent_states,
            goal=goal_states,
            obstacle=obstacles,
        )

        info = {}
        if get_eval_info:
            info["inside_obstacles"] = inside_obstacles(
                agent_states[:, :3],
                obstacles,
                r=self._params["drone_radius"],
            )
            info["applied_action"] = applied_action

        return (
            self.get_graph(next_env_state),
            reward,
            cost,
            done,
            info,
        )

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------

    def project_action(self, action: Action) -> Action:
        """
        Apply componentwise action limits and, optionally, an L2 norm limit.
        """
        action = self.clip_action(action)

        if not self._params["use_norm_limit"]:
            return action

        acc_norm = jnp.linalg.norm(
            action,
            axis=-1,
            keepdims=True,
        )
        scale = jnp.minimum(
            1.0,
            self.ma / jnp.maximum(acc_norm, 1e-6),
        )
        return action * scale

    @ft.partial(jax.jit, static_argnums=(0,))
    def agent_dynamics(
        self,
        x: AgentState,
        u: Action,
    ) -> AgentState:
        """
        Continuous-time dynamics for one agent.

        x = [px, py, pz, vx, vy, vz]
        u = [ax, ay, az]
        """
        assert x.shape == (self.state_dim,)
        assert u.shape == (self.action_dim,)

        acceleration = self.project_action(u)
        velocity = x[3:6]

        return jnp.concatenate(
            [velocity, acceleration],
            axis=0,
        )

    def agent_xdot(
        self,
        agent_states: AgentState,
        action: Action,
    ) -> AgentState:
        """
        Continuous-time dynamics for one agent or a batch of agents.
        """
        if agent_states.ndim == 1:
            return self.agent_dynamics(agent_states, action)

        assert agent_states.shape == (
            self.num_agents,
            self.state_dim,
        )
        assert action.shape == (
            self.num_agents,
            self.action_dim,
        )

        return jax_vmap(self.agent_dynamics)(
            agent_states,
            action,
        )

    def agent_step_exact(
        self,
        agent_states: AgentState,
        action: Action,
    ) -> AgentState:
        """
        Exact discrete update for constant acceleration over one step.

            p_next = p + v dt + 0.5 a dt^2
            v_next = v + a dt
        """
        assert agent_states.shape == (
            self.num_agents,
            self.state_dim,
        )
        assert action.shape == (
            self.num_agents,
            self.action_dim,
        )

        acceleration = self.project_action(action)

        position = agent_states[:, :3]
        velocity = agent_states[:, 3:6]

        next_position = (
            position
            + velocity * self.dt
            + 0.5 * acceleration * self.dt**2
        )
        next_velocity = velocity + acceleration * self.dt

        next_state = jnp.concatenate(
            [next_position, next_velocity],
            axis=1,
        )
        return self.clip_state(next_state)

    def agent_step_euler(
        self,
        agent_states: AgentState,
        action: Action,
    ) -> AgentState:
        """
        Forward-Euler alternative retained for compatibility/testing.
        """
        x_dot = self.agent_xdot(agent_states, action)
        return self.clip_state(
            agent_states + self.dt * x_dot
        )

    def control_affine_dyn_single(
        self,
        state: AgentState,
    ) -> Tuple[Array, Array]:
        """
        Unsaturated control-affine dynamics for one agent.

            x_dot = f(x) + g(x)u

        The norm projection is not included in g(x). It is applied to the
        action before simulation.
        """
        assert state.shape == (self.state_dim,)

        f = jnp.concatenate(
            [
                state[3:6],
                jnp.zeros((3,)),
            ],
            axis=0,
        )

        g = jnp.concatenate(
            [
                jnp.zeros((3, 3)),
                jnp.eye(3),
            ],
            axis=0,
        )

        return f, g

    def control_affine_dyn(
        self,
        state: State,
    ) -> Tuple[Array, Array]:
        assert state.ndim == 2

        f, g = jax_vmap(
            self.control_affine_dyn_single
        )(state)

        assert f.shape == state.shape
        assert g.shape == (
            state.shape[0],
            self.state_dim,
            self.action_dim,
        )
        return f, g

    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = -self.ma * jnp.ones(
            (self.action_dim,)
        )
        upper_lim = self.ma * jnp.ones(
            (self.action_dim,)
        )
        return lower_lim, upper_lim

    def state_lim(
        self,
        state: Optional[State] = None,
    ) -> Tuple[State, State]:
        max_velocity = self._params["max_velocity"]

        lower_lim = jnp.array(
            [
                -jnp.inf,
                -jnp.inf,
                -jnp.inf,
                -max_velocity,
                -max_velocity,
                -max_velocity,
            ]
        )
        upper_lim = jnp.array(
            [
                jnp.inf,
                jnp.inf,
                jnp.inf,
                max_velocity,
                max_velocity,
                max_velocity,
            ]
        )
        return lower_lim, upper_lim

    # ------------------------------------------------------------------
    # Nominal goal controller
    # ------------------------------------------------------------------

    def u_ref(self, graph: GraphsTuple) -> Action:
        agent = graph.type_states(
            type_idx=self.AGENT,
            n_type=self.num_agents,
        )
        goal = graph.type_states(
            type_idx=self.GOAL,
            n_type=self.num_agents,
        )

        position_error = goal[:, :3] - agent[:, :3]
        distance = jnp.linalg.norm(
            position_error,
            axis=1,
            keepdims=True,
        )

        position_scale = jnp.minimum(
            1.0,
            self.comm_radius
            / jnp.maximum(distance, 1e-6),
        )
        position_error = position_error * position_scale

        velocity_error = goal[:, 3:6] - agent[:, 3:6]
        error = jnp.concatenate(
            [position_error, velocity_error],
            axis=1,
        )

        action = error @ self._K.T
        return self.project_action(action)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def edge_state(self, state: State) -> State:
        """
        Position and velocity are already expressed in the world frame.
        """
        assert state.shape[-1] == self.state_dim
        return state

    def add_edge_feats(
        self,
        graph: GraphsTuple,
        state: State,
    ) -> GraphsTuple:
        assert graph.is_single
        assert state.ndim == 2

        edge_state = self.edge_state(state)
        edge_feats = (
            edge_state[graph.receivers]
            - edge_state[graph.senders]
        )

        # Clip only relative position, not relative velocity.
        position_norm = jnp.sqrt(
            1e-6
            + jnp.sum(
                edge_feats[:, :3] ** 2,
                axis=-1,
                keepdims=True,
            )
        )
        safe_norm = jnp.maximum(
            position_norm,
            self.comm_radius,
        )
        scale = jnp.where(
            position_norm > self.comm_radius,
            self.comm_radius / safe_norm,
            1.0,
        )

        edge_feats = edge_feats.at[:, :3].set(
            edge_feats[:, :3] * scale
        )

        return graph._replace(
            edges=edge_feats,
            states=state,
        )

    def edge_blocks(
        self,
        state: EnvState,
        lidar_data: Pos3d,
    ) -> list[EdgeBlock]:
        n_hits = self.num_agents * self.n_rays

        agent_position = state.agent[:, :3]
        agent_edge_state = self.edge_state(state.agent)

        # --------------------------------------------------------------
        # Agent-agent communication edges
        # --------------------------------------------------------------
        position_difference = (
            agent_position[:, None, :]
            - agent_position[None, :, :]
        )
        distance = jnp.linalg.norm(
            position_difference,
            axis=-1,
        )
        distance += jnp.eye(self.num_agents) * (
            self.comm_radius + 1.0
        )

        state_difference = (
            agent_edge_state[:, None, :]
            - agent_edge_state[None, :, :]
        )
        agent_agent_mask = distance < self.comm_radius

        id_agent = jnp.arange(self.num_agents)

        agent_agent_edges = EdgeBlock(
            state_difference,
            agent_agent_mask,
            id_agent,
            id_agent,
        )

        # --------------------------------------------------------------
        # Permanent one-to-one agent-goal edges
        # --------------------------------------------------------------
        id_goal = jnp.arange(
            self.num_agents,
            2 * self.num_agents,
        )
        agent_goal_mask = jnp.eye(
            self.num_agents,
            dtype=bool,
        )

        goal_edge_state = self.edge_state(state.goal)
        agent_goal_features = (
            agent_edge_state[:, None, :]
            - goal_edge_state[None, :, :]
        )

        goal_position_norm = jnp.sqrt(
            1e-6
            + jnp.sum(
                agent_goal_features[..., :3] ** 2,
                axis=-1,
                keepdims=True,
            )
        )
        safe_goal_norm = jnp.maximum(
            goal_position_norm,
            self.comm_radius,
        )
        goal_scale = jnp.where(
            goal_position_norm > self.comm_radius,
            self.comm_radius / safe_goal_norm,
            1.0,
        )

        agent_goal_features = (
            agent_goal_features.at[..., :3].set(
                agent_goal_features[..., :3]
                * goal_scale
            )
        )

        agent_goal_edges = EdgeBlock(
            agent_goal_features,
            agent_goal_mask,
            id_agent,
            id_goal,
        )

        # --------------------------------------------------------------
        # Agent-LiDAR hit edges
        # --------------------------------------------------------------
        id_obs = jnp.arange(
            2 * self.num_agents,
            2 * self.num_agents + n_hits,
        )
        lidar_edge_state = self.edge_state(lidar_data)

        agent_obs_edges = []

        for i in range(self.num_agents):
            id_hits = jnp.arange(
                i * self.n_rays,
                (i + 1) * self.n_rays,
            )

            lidar_position = (
                agent_position[i]
                - lidar_data[id_hits, :3]
            )
            lidar_features = (
                agent_edge_state[i]
                - lidar_edge_state[id_hits]
            )
            lidar_distance = jnp.linalg.norm(
                lidar_position,
                axis=-1,
            )

            active_lidar = (
                lidar_distance
                < jnp.maximum(
                    self.comm_radius - 1e-1,
                    0.0,
                )
            )
            agent_obs_mask = active_lidar[None, :]

            agent_obs_edges.append(
                EdgeBlock(
                    lidar_features[None, :, :],
                    agent_obs_mask,
                    id_agent[i][None],
                    id_obs[id_hits],
                )
            )

        return [
            agent_agent_edges,
            agent_goal_edges,
            *agent_obs_edges,
        ]

    def get_graph(
        self,
        state: EnvState,
        adjacency: Array = None,
    ) -> GraphsTuple:
        n_hits = self.n_rays * self.num_agents
        n_nodes = 2 * self.num_agents + n_hits

        # Node one-hot features:
        # obstacle = [1, 0, 0]
        # goal     = [0, 1, 0]
        # agent    = [0, 0, 1]
        node_features = jnp.zeros(
            (n_nodes, self.node_dim)
        )
        node_features = node_features.at[
            :self.num_agents,
            2,
        ].set(1.0)
        node_features = node_features.at[
            self.num_agents:2 * self.num_agents,
            1,
        ].set(1.0)
        node_features = node_features.at[
            2 * self.num_agents:,
            0,
        ].set(1.0)

        node_type = jnp.full(
            (n_nodes,),
            self.AGENT,
            dtype=jnp.int32,
        )
        node_type = node_type.at[
            self.num_agents:2 * self.num_agents
        ].set(self.GOAL)
        node_type = node_type.at[
            2 * self.num_agents:
        ].set(self.OBS)

        get_lidar_vmap = jax.vmap(
            ft.partial(
                get_lidar,
                obstacles=state.obstacle,
                num_beams=self._params["n_rays"],
                sense_range=self.comm_radius,
                max_returns=self.n_rays,
            )
        )

        lidar_positions = merge01(
            get_lidar_vmap(state.agent[:, :3])
        )

        # Static obstacle hit points have zero velocity.
        lidar_data = jnp.concatenate(
            [
                lidar_positions,
                jnp.zeros(
                    (lidar_positions.shape[0], 3)
                ),
            ],
            axis=-1,
        )

        edge_blocks = self.edge_blocks(
            state,
            lidar_data,
        )

        graph = GetGraph(
            nodes=node_features,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state,
            states=jnp.concatenate(
                [
                    state.agent,
                    state.goal,
                    lidar_data,
                ],
                axis=0,
            ),
        ).to_padded()

        # Optional desired agent-agent connectivity matrix. This supports the
        # custom GraphsTuple field used in the user's code while remaining
        # compatible with the original GCBF+ GraphsTuple.
        if (
            adjacency is not None
            and hasattr(graph, "connectivity")
        ):
            assert adjacency.shape == (
                self.num_agents,
                self.num_agents,
            )

            connectivity = jnp.zeros(
                (
                    graph.nodes.shape[0],
                    graph.nodes.shape[0],
                )
            )
            connectivity = connectivity.at[
                :self.num_agents,
                :self.num_agents,
            ].set(adjacency)

            graph = graph._replace(
                connectivity=connectivity
            )

        return graph

    def forward_graph(
        self,
        graph: GraphsTuple,
        action: Action,
    ) -> GraphsTuple:
        agent_states = graph.type_states(
            type_idx=self.AGENT,
            n_type=self.num_agents,
        )
        goal_states = graph.type_states(
            type_idx=self.GOAL,
            n_type=self.num_agents,
        )
        obstacle_states = graph.type_states(
            type_idx=self.OBS,
            n_type=self.n_rays * self.num_agents,
        )

        applied_action = self.project_action(action)
        next_agent_states = self.agent_step_exact(
            agent_states,
            applied_action,
        )

        next_states = jnp.concatenate(
            [
                next_agent_states,
                goal_states,
                obstacle_states,
            ],
            axis=0,
        )

        return self.add_edge_feats(
            graph,
            next_states,
        )

    # ------------------------------------------------------------------
    # Safety, collision, and task masks
    # ------------------------------------------------------------------

    def get_cost(
        self,
        graph: EnvGraphsTuple,
    ) -> Cost:
        agent_position = graph.type_states(
            type_idx=self.AGENT,
            n_type=self.num_agents,
        )[:, :3]

        position_difference = (
            agent_position[:, None, :]
            - agent_position[None, :, :]
        )
        distance = jnp.linalg.norm(
            position_difference,
            axis=-1,
        )
        distance += jnp.eye(self.num_agents) * 1e6

        agent_collision = (
            distance
            < 2.0 * self._params["drone_radius"]
        ).any(axis=1)

        obstacle_collision = inside_obstacles(
            agent_position,
            graph.env_states.obstacle,
            r=self._params["drone_radius"],
        )

        return (
            agent_collision.mean()
            + obstacle_collision.mean()
        )

    @ft.partial(jax.jit, static_argnums=(0,))
    def safe_mask(
        self,
        graph: GraphsTuple,
    ) -> Array:
        agent_position = graph.type_states(
            type_idx=self.AGENT,
            n_type=self.num_agents,
        )[:, :3]

        position_difference = (
            agent_position[:, None, :]
            - agent_position[None, :, :]
        )
        distance = jnp.linalg.norm(
            position_difference,
            axis=-1,
        )
        distance += jnp.eye(self.num_agents) * (
            4.0 * self._params["drone_radius"]
            + 1.0
        )

        safe_agent = jnp.all(
            distance
            > 4.0 * self._params["drone_radius"],
            axis=1,
        )

        safe_obstacle = jnp.logical_not(
            inside_obstacles(
                agent_position,
                graph.env_states.obstacle,
                2.0 * self._params["drone_radius"],
            )
        )

        return jnp.logical_and(
            safe_agent,
            safe_obstacle,
        )

    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_mask(
        self,
        graph: GraphsTuple,
    ) -> Array:
        agent_position = graph.type_states(
            type_idx=self.AGENT,
            n_type=self.num_agents,
        )[:, :3]

        position_difference = (
            agent_position[:, None, :]
            - agent_position[None, :, :]
        )
        distance = jnp.linalg.norm(
            position_difference,
            axis=-1,
        )
        distance += jnp.eye(self.num_agents) * (
            2.5 * self._params["drone_radius"]
            + 1.0
        )

        unsafe_agent = jnp.any(
            distance
            < 2.5 * self._params["drone_radius"],
            axis=1,
        )

        unsafe_obstacle = inside_obstacles(
            agent_position,
            graph.env_states.obstacle,
            1.5 * self._params["drone_radius"],
        )

        return jnp.logical_or(
            unsafe_agent,
            unsafe_obstacle,
        )

    def collision_mask(
        self,
        graph: GraphsTuple,
    ) -> Array:
        agent_position = graph.type_states(
            type_idx=self.AGENT,
            n_type=self.num_agents,
        )[:, :3]

        position_difference = (
            agent_position[:, None, :]
            - agent_position[None, :, :]
        )
        distance = jnp.linalg.norm(
            position_difference,
            axis=-1,
        )
        distance += jnp.eye(self.num_agents) * (
            2.0 * self._params["drone_radius"]
            + 1.0
        )

        agent_collision = jnp.any(
            distance
            < 2.0 * self._params["drone_radius"],
            axis=1,
        )

        obstacle_collision = inside_obstacles(
            agent_position,
            graph.env_states.obstacle,
            self._params["drone_radius"],
        )

        return jnp.logical_or(
            agent_collision,
            obstacle_collision,
        )

    def finish_mask(
        self,
        graph: GraphsTuple,
    ) -> Array:
        agent_position = graph.type_states(
            type_idx=self.AGENT,
            n_type=self.num_agents,
        )[:, :3]
        goal_position = graph.env_states.goal[:, :3]

        return (
            jnp.linalg.norm(
                agent_position - goal_position,
                axis=1,
            )
            < 2.0 * self._params["drone_radius"]
        )

    # ------------------------------------------------------------------
    # Visualization metadata
    # ------------------------------------------------------------------

    def render_video(
        self,
        rollout: RolloutResult,
        video_path: pathlib.Path,
        Ta_is_unsafe=None,
        viz_opts: dict = None,
        dpi: int = 100,
        **kwargs,
    ) -> None:
        render_video(
            rollout=rollout,
            video_path=video_path,
            side_length=self.area_size,
            dim=3,
            n_agent=self.num_agents,
            n_rays=self.n_rays,
            r=self._params["drone_radius"],
            Ta_is_unsafe=Ta_is_unsafe,
            viz_opts=viz_opts,
            dpi=dpi,
            **kwargs,
        )

    @property
    def x_labels(self):
        labels = [
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
        ]
        assert len(labels) == self.state_dim
        return labels

    @property
    def uhl_labels(self):
        labels = [
            "ax",
            "ay",
            "az",
        ]
        assert len(labels) == self.action_dim
        return labels
