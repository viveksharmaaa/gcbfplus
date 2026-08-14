import functools as ft
import pathlib

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from typing import NamedTuple, Optional, Tuple

from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, AgentState, Array, Cost, Done, Info, Pos3d, Reward, State
from ..utils.utils import merge01, jax_vmap
from .base import MultiAgentEnv, RolloutResult
from .linear_drone import LinearDrone
from .obstacle import Obstacle, Sphere
from .plot import render_video
from .utils import RK4_step, get_lidar, inside_obstacles, get_node_goal_rng


class Quadcopter(MultiAgentEnv):
    """
    Reduced-order quadrotor environment for GCBF+.

    The class name is intentionally kept as ``CrazyFlie`` so that the remaining
    GCBF+ configuration/registry code does not need to be changed.

    State of each agent:
        x = [px, py, pz, vx, vy, vz, f, phi, theta, psi]

    Control:
        u = [f_dot, phi_dot, theta_dot, psi_dot]

    Continuous-time dynamics:
        p_dot_x = vx
        p_dot_y = vy
        p_dot_z = vz
        v_dot_x = -f sin(theta)
        v_dot_y =  f cos(theta) sin(phi)
        v_dot_z =  g - f cos(theta) cos(phi)
        f_dot     = u[0]
        phi_dot   = u[1]
        theta_dot = u[2]
        psi_dot   = u[3]

    Notes
    -----
    1. This model uses a z-down convention, exactly as implied by
       v_dot_z = g - f cos(theta) cos(phi). Hover is f = g.
    2. Actions are normalized only by their explicit bounds [-1, 1].
       Thus, every component is a physical rate limited to magnitude 1.
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
        "obs_len_range": [0.1, 0.6],
        "n_obs": 0,
        "g": 9.81,

        # Nominal-controller gains. They can be overridden through params.
        "kp_pos": 0.8,
        "kd_vel": 1.2,
        "k_f": 1.5,
        "k_att": 2.0,
        "max_acc_xy": 3.0,
        "max_acc_z": 3.0,
    }

    # State indices:
    # [px, py, pz, vx, vy, vz, f, phi, theta, psi]
    PX, PY, PZ, VX, VY, VZ, F, PHI, THETA, PSI = range(10)

    # Control indices:
    # [f_dot, phi_dot, theta_dot, psi_dot]
    F_DOT, PHI_DOT, THETA_DOT, PSI_DOT = range(4)

    def __init__(
        self,
        num_agents: int,
        area_size: float,
        max_step: int = 256,
        max_travel: float = None,
        dt: float = 0.03,
        params: dict = None,
    ):
        super().__init__(num_agents, area_size, max_step, max_travel, dt, params)

        self.create_obstacles = jax.vmap(Sphere.create)
        self.n_rays = min(16, self._params["n_rays"] ** 2 // 2 + 2)

    @property
    def state_dim(self) -> int:
        return 10

    @property
    def node_dim(self) -> int:
        # One-hot node type: obstacle, goal, agent.
        return 3

    @property
    def edge_dim(self) -> int:
        # Relative [position(3), velocity(3), f, phi, theta, psi].
        return 10

    @property
    def action_dim(self) -> int:
        return 4

    def reset(self, key: Array) -> GraphsTuple:
        self._t = 0

        # Random spherical obstacles.
        n_rng_obs = self._params["n_obs"]
        assert n_rng_obs >= 0

        obstacle_key, key = jr.split(key)
        obs_pos = jr.uniform(
            obstacle_key,
            (n_rng_obs, 3),
            minval=0.0,
            maxval=self.area_size,
        )

        radius_key, key = jr.split(key)
        obs_radius = jr.uniform(
            radius_key,
            (n_rng_obs,),
            minval=self._params["obs_len_range"][0] / 2.0,
            maxval=self._params["obs_len_range"][1] / 2.0,
        )
        obstacles = self.create_obstacles(obs_pos, obs_radius)

        # Sample collision-free initial positions and goals.
        positions, goal_positions = get_node_goal_rng(
            key,
            self.area_size,
            3,
            obstacles,
            self.num_agents,
            4.0 * self.params["drone_radius"],
            self.max_travel,
        )

        g = self._params["g"]

        # Initial state: zero velocity, hover thrust, zero attitude.
        states = jnp.concatenate(
            [
                positions,
                jnp.zeros((self.num_agents, 3)),       # vx, vy, vz
                g * jnp.ones((self.num_agents, 1)),    # f = g
                jnp.zeros((self.num_agents, 3)),       # phi, theta, psi
            ],
            axis=1,
        )

        # Goal states use hover equilibrium.
        goals = jnp.concatenate(
            [
                goal_positions,
                jnp.zeros((self.num_agents, 3)),
                g * jnp.ones((self.num_agents, 1)),
                jnp.zeros((self.num_agents, 3)),
            ],
            axis=1,
        )

        env_state = self.EnvState(states, goals, obstacles)
        return self.get_graph(env_state)

    def step(
        self,
        graph: EnvGraphsTuple,
        action: Action,
        get_eval_info: bool = False,
    ) -> Tuple[EnvGraphsTuple, Reward, Cost, Done, Info]:
        self._t += 1

        agent_states = graph.type_states(type_idx=self.AGENT, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=self.GOAL, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle

        action = self.clip_action(action)
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        next_agent_states = self.agent_step_rk4(agent_states, action)

        done = jnp.array(False)
        reward = -jnp.mean(jnp.sum((action - self.u_ref(graph)) ** 2, axis=1))
        cost = self.get_cost(graph)

        next_state = self.EnvState(next_agent_states, goal_states, obstacles)

        info = {}
        if get_eval_info:
            info["inside_obstacles"] = inside_obstacles(
                agent_states[:, :3],
                obstacles,
                r=self._params["drone_radius"],
            )

        return self.get_graph(next_state), reward, cost, done, info

    def get_cost(self, graph: EnvGraphsTuple) -> Cost:
        agent_pos = graph.type_states(
            type_idx=self.AGENT,
            n_type=self.num_agents,
        )[:, :3]
        obstacles = graph.env_states.obstacle

        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6

        agent_collision = (
            dist < 2.0 * self._params["drone_radius"]
        ).any(axis=1)

        obstacle_collision = inside_obstacles(
            agent_pos,
            obstacles,
            r=self._params["drone_radius"],
        )

        return agent_collision.mean() + obstacle_collision.mean()

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
            r=self.params["drone_radius"],
            Ta_is_unsafe=Ta_is_unsafe,
            viz_opts=viz_opts,
            dpi=dpi,
            **kwargs,
        )

    def edge_state(self, state: State) -> State:
        """
        The reduced model already stores position and velocity in the world frame,
        so no body-to-world transformation is needed.
        """
        assert state.shape[-1] == self.state_dim
        return state

    def add_edge_feats(self, graph: GraphsTuple, state: State) -> GraphsTuple:
        assert graph.is_single
        assert state.ndim == 2

        edge_state = self.edge_state(state)
        edge_feats = edge_state[graph.receivers] - edge_state[graph.senders]

        # Clip only relative position features at the communication radius.
        pos_norm = jnp.sqrt(
            1e-6 + jnp.sum(edge_feats[:, :3] ** 2, axis=-1, keepdims=True)
        )
        comm_radius = self._params["comm_radius"]
        safe_norm = jnp.maximum(pos_norm, comm_radius)
        coef = jnp.where(pos_norm > comm_radius, comm_radius / safe_norm, 1.0)
        edge_feats = edge_feats.at[:, :3].set(edge_feats[:, :3] * coef)

        return graph._replace(edges=edge_feats, states=state)

    def edge_blocks(self, state: EnvState, lidar_data: Pos3d) -> list[EdgeBlock]:
        n_hits = self.num_agents * self.n_rays

        agent_pos = state.agent[:, :3]
        agent_edge_state = self.edge_state(state.agent)

        # Agent-agent edges.
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(self.num_agents) * (self._params["comm_radius"] + 1.0)

        state_diff = (
            agent_edge_state[:, None, :] - agent_edge_state[None, :, :]
        )
        agent_agent_mask = dist < self._params["comm_radius"]

        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(
            state_diff,
            agent_agent_mask,
            id_agent,
            id_agent,
        )

        # Permanent agent-goal edges.
        id_goal = jnp.arange(self.num_agents, 2 * self.num_agents)
        agent_goal_mask = jnp.eye(self.num_agents, dtype=bool)

        goal_edge_state = self.edge_state(state.goal)
        agent_goal_feats = (
            agent_edge_state[:, None, :] - goal_edge_state[None, :, :]
        )

        pos_norm = jnp.sqrt(
            1e-6 + jnp.sum(agent_goal_feats[:, :, :3] ** 2, axis=-1, keepdims=True)
        )
        comm_radius = self._params["comm_radius"]
        safe_norm = jnp.maximum(pos_norm, comm_radius)
        coef = jnp.where(pos_norm > comm_radius, comm_radius / safe_norm, 1.0)
        agent_goal_feats = agent_goal_feats.at[:, :, :3].set(
            agent_goal_feats[:, :, :3] * coef
        )

        agent_goal_edges = EdgeBlock(
            agent_goal_feats,
            agent_goal_mask,
            id_agent,
            id_goal,
        )

        # Agent-obstacle/LiDAR edges.
        id_obs = jnp.arange(2 * self.num_agents, 2 * self.num_agents + n_hits)
        lidar_edge_state = self.edge_state(lidar_data)
        agent_obs_edges = []

        for i in range(self.num_agents):
            id_hits = jnp.arange(i * self.n_rays, (i + 1) * self.n_rays)

            lidar_pos = agent_pos[i] - lidar_data[id_hits, :3]
            lidar_feats = agent_edge_state[i] - lidar_edge_state[id_hits]
            lidar_dist = jnp.linalg.norm(lidar_pos, axis=-1)

            active_lidar = (
                lidar_dist < self._params["comm_radius"] - 1e-1
            )
            agent_obs_mask = active_lidar[None, :]

            agent_obs_edges.append(
                EdgeBlock(
                    lidar_feats[None, :, :],
                    agent_obs_mask,
                    id_agent[i][None],
                    id_obs[id_hits],
                )
            )

        return [agent_agent_edges, agent_goal_edges] + agent_obs_edges

    def _single_agent_f(self, x: Array) -> Array:
        """Drift vector f(x) in x_dot = f(x) + g(x)u."""
        assert x.shape == (self.state_dim,)

        vx, vy, vz = x[self.VX], x[self.VY], x[self.VZ]
        thrust = x[self.F]
        phi, theta = x[self.PHI], x[self.THETA]
        gravity = self._params["g"]

        return jnp.array(
            [
                vx,
                vy,
                vz,
                -thrust * jnp.sin(theta),
                thrust * jnp.cos(theta) * jnp.sin(phi),
                gravity - thrust * jnp.cos(theta) * jnp.cos(phi),
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

    def _single_agent_g(self, x: Array) -> Array:
        """Input matrix g(x); it is constant for this reduced-order model."""
        assert x.shape == (self.state_dim,)

        g_mat = jnp.zeros((self.state_dim, self.action_dim))
        g_mat = g_mat.at[self.F, self.F_DOT].set(1.0)
        g_mat = g_mat.at[self.PHI, self.PHI_DOT].set(1.0)
        g_mat = g_mat.at[self.THETA, self.THETA_DOT].set(1.0)
        g_mat = g_mat.at[self.PSI, self.PSI_DOT].set(1.0)
        return g_mat

    def _agent_xdot_single_agent(
        self,
        state: AgentState,
        action: Action,
    ) -> AgentState:
        assert state.shape == (self.state_dim,)
        assert action.shape == (self.action_dim,)

        xdot = self._single_agent_f(state) + self._single_agent_g(state) @ action
        assert xdot.shape == (self.state_dim,)
        return xdot

    def agent_xdot(
        self,
        agent_states: AgentState,
        actions: Action,
    ) -> AgentState:
        assert actions.ndim == agent_states.ndim

        if actions.ndim == 1:
            return self._agent_xdot_single_agent(agent_states, actions)

        return jax_vmap(self._agent_xdot_single_agent)(agent_states, actions)

    def agent_step_rk4(
        self,
        agent_state: Array,
        action: Array,
    ) -> Array:
        assert agent_state.shape == (self.num_agents, self.state_dim)
        assert action.shape == (self.num_agents, self.action_dim)

        next_state = RK4_step(
            self.agent_xdot,
            agent_state,
            action,
            self.dt,
        )
        return self.clip_state(next_state)

    def rk4_single(
        self,
        agent_state: Array,
        action: Array,
    ) -> Tuple[Array, Array]:
        assert agent_state.shape == (self.state_dim,)
        assert action.shape == (self.action_dim,)

        next_state = RK4_step(
            self._agent_xdot_single_agent,
            agent_state,
            action,
            self.dt,
        )
        return self.clip_state(next_state), self.clip_action(action)

    def control_affine_dyn_single(
        self,
        state: AgentState,
    ) -> Tuple[Array, Array]:
        return self._single_agent_f(state), self._single_agent_g(state)

    def control_affine_dyn(
        self,
        state: State,
    ) -> Tuple[Array, Array]:
        assert state.ndim == 2
        return jax_vmap(self.control_affine_dyn_single)(state)

    def action_lim(self) -> Tuple[Action, Action]:
        low_lim = -jnp.ones((self.action_dim,))
        up_lim = jnp.ones((self.action_dim,))
        return low_lim, up_lim

    def state_lim(
        self,
        state: Optional[State] = None,
    ) -> Tuple[State, State]:
        g = self._params["g"]

        # Paper bounds:
        # p in [-30,30]^3, v in [-1.5,1.5]^3,
        # f in [0.5g,2g], angles in [-pi/3,pi/3].
        low_lim = jnp.array(
            [
                -30.0, -30.0, -30.0,
                -1.5, -1.5, -1.5,
                0.5 * g,
                -jnp.pi / 3.0,
                -jnp.pi / 3.0,
                -jnp.pi / 3.0,
            ]
        )
        up_lim = jnp.array(
            [
                30.0, 30.0, 30.0,
                1.5, 1.5, 1.5,
                2.0 * g,
                jnp.pi / 3.0,
                jnp.pi / 3.0,
                jnp.pi / 3.0,
            ]
        )
        return low_lim, up_lim

    def u_ref_inner_single(
        self,
        state: AgentState,
        goal: AgentState,
    ) -> Action:
        """
        Nominal goal-seeking controller.

        First compute a desired translational acceleration using PD feedback.
        Then map that acceleration to desired thrust, roll, and pitch under the
        same reduced-order dynamics. Finally command their rates.

        Near hover:
            ax ≈ -g theta
            ay ≈  g phi
            az ≈  g - f
        """
        pos_error = state[:3] - goal[:3]
        vel_error = state[3:6] - goal[3:6]

        # Avoid very large commands for distant goals.
        dist = jnp.linalg.norm(pos_error)
        dist_safe = jnp.maximum(dist, 1e-6)
        coef = jnp.where(
            dist > self.comm_radius,
            self.comm_radius / dist_safe,
            1.0,
        )
        pos_error = pos_error * coef

        kp = self._params["kp_pos"]
        kd = self._params["kd_vel"]

        acc_des = -kp * pos_error - kd * vel_error
        acc_des = acc_des.at[:2].set(
            jnp.clip(
                acc_des[:2],
                -self._params["max_acc_xy"],
                self._params["max_acc_xy"],
            )
        )
        acc_des = acc_des.at[2].set(
            jnp.clip(
                acc_des[2],
                -self._params["max_acc_z"],
                self._params["max_acc_z"],
            )
        )

        gravity = self._params["g"]

        # Exact inversion of the reduced translational model.
        # Let b = [ax, ay, g-az] = f[-sin(theta), cos(theta)sin(phi),
        # cos(theta)cos(phi)].
        b_x = acc_des[0]
        b_y = acc_des[1]
        b_z = gravity - acc_des[2]

        f_des = jnp.sqrt(b_x**2 + b_y**2 + b_z**2)
        f_des = jnp.clip(f_des, 0.5 * gravity, 2.0 * gravity)

        theta_des = -jnp.arcsin(
            jnp.clip(b_x / jnp.maximum(f_des, 1e-6), -1.0, 1.0)
        )
        phi_des = jnp.arctan2(b_y, b_z)

        theta_des = jnp.clip(theta_des, -jnp.pi / 3.0, jnp.pi / 3.0)
        phi_des = jnp.clip(phi_des, -jnp.pi / 3.0, jnp.pi / 3.0)

        # Keep the goal yaw, normally zero in reset().
        psi_des = goal[self.PSI]

        k_f = self._params["k_f"]
        k_att = self._params["k_att"]

        action = jnp.array(
            [
                k_f * (f_des - state[self.F]),
                k_att * (phi_des - state[self.PHI]),
                k_att * (theta_des - state[self.THETA]),
                k_att * (psi_des - state[self.PSI]),
            ]
        )
        return self.clip_action(action)

    def u_ref(self, graph: GraphsTuple) -> Action:
        agent = graph.type_states(
            type_idx=self.AGENT,
            n_type=self.num_agents,
        )
        goal = graph.type_states(
            type_idx=self.GOAL,
            n_type=self.num_agents,
        )
        return jax_vmap(self.u_ref_inner_single)(agent, goal)

    def render(self, graph: GraphsTuple) -> plt.Figure:
        raise NotImplementedError("Use render_video() for this 3-D environment.")

    def get_graph(self, state: EnvState) -> GraphsTuple:
        n_hits = self.n_rays * self.num_agents
        n_nodes = 2 * self.num_agents + n_hits

        node_feats = jnp.zeros((n_nodes, self.node_dim))
        node_feats = node_feats.at[:self.num_agents, 2].set(1.0)
        node_feats = node_feats.at[
            self.num_agents:2 * self.num_agents, 1
        ].set(1.0)
        node_feats = node_feats.at[-n_hits:, 0].set(1.0)

        node_type = jnp.zeros(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[
            self.num_agents:2 * self.num_agents
        ].set(LinearDrone.GOAL)
        node_type = node_type.at[-n_hits:].set(LinearDrone.OBS)

        get_lidar_vmap = jax.vmap(
            ft.partial(
                get_lidar,
                obstacles=state.obstacle,
                num_beams=self.params["n_rays"],
                sense_range=self._params["comm_radius"],
                max_returns=self.n_rays,
            )
        )
        lidar_positions = merge01(get_lidar_vmap(state.agent[:, :3]))

        # Represent static LiDAR hit points as zero-velocity, hover-equilibrium
        # pseudo-states. Only relative position is physically relevant.
        gravity = self._params["g"]
        lidar_data = jnp.concatenate(
            [
                lidar_positions,
                jnp.zeros((lidar_positions.shape[0], 3)),
                gravity * jnp.ones((lidar_positions.shape[0], 1)),
                jnp.zeros((lidar_positions.shape[0], 3)),
            ],
            axis=-1,
        )

        edge_blocks = self.edge_blocks(state, lidar_data)

        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state,
            states=jnp.concatenate(
                [state.agent, state.goal, lidar_data],
                axis=0,
            ),
        ).to_padded()

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
        obs_states = graph.type_states(
            type_idx=self.OBS,
            n_type=self._params["n_rays"] * self.num_agents,
        )

        action = self.clip_action(action)
        next_agent_states = self.agent_step_rk4(agent_states, action)

        next_states = jnp.concatenate(
            [next_agent_states, goal_states, obs_states],
            axis=0,
        )
        return self.add_edge_feats(graph, next_states)

    def safe_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(
            type_idx=self.AGENT,
            n_type=self.num_agents,
        )[:, :3]

        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(self.num_agents) * (
            4.0 * self._params["drone_radius"] + 1.0
        )

        safe_agent = jnp.min(
            dist > 4.0 * self._params["drone_radius"],
            axis=1,
        )
        safe_obs = jnp.logical_not(
            inside_obstacles(
                agent_pos,
                graph.env_states.obstacle,
                2.0 * self._params["drone_radius"],
            )
        )
        return jnp.logical_and(safe_agent, safe_obs)

    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(
            type_idx=self.AGENT,
            n_type=self.num_agents,
        )[:, :3]

        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(self.num_agents) * (
            2.5 * self._params["drone_radius"] + 1.0
        )

        unsafe_agent = jnp.max(
            dist < 2.5 * self._params["drone_radius"],
            axis=1,
        )
        unsafe_obs = inside_obstacles(
            agent_pos,
            graph.env_states.obstacle,
            1.5 * self._params["drone_radius"],
        )
        return jnp.logical_or(unsafe_agent, unsafe_obs)

    def collision_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(
            type_idx=self.AGENT,
            n_type=self.num_agents,
        )[:, :3]

        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(self.num_agents) * (
            2.0 * self._params["drone_radius"] + 1.0
        )

        unsafe_agent = jnp.max(
            dist < 2.0 * self._params["drone_radius"],
            axis=1,
        )
        unsafe_obs = inside_obstacles(
            agent_pos,
            graph.env_states.obstacle,
            self._params["drone_radius"],
        )
        return jnp.logical_or(unsafe_agent, unsafe_obs)

    def finish_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(
            type_idx=self.AGENT,
            n_type=self.num_agents,
        )[:, :3]
        goal_pos = graph.env_states.goal[:, :3]

        return (
            jnp.linalg.norm(agent_pos - goal_pos, axis=1)
            < 2.0 * self._params["drone_radius"]
        )

    @property
    def comm_radius(self):
        return self._params["comm_radius"]

    @property
    def x_labels(self):
        labels = [
            "x", "y", "z",
            "vx", "vy", "vz",
            "f",
            r"$\phi$", r"$\theta$", r"$\psi$",
        ]
        assert len(labels) == self.state_dim
        return labels

    @property
    def uhl_labels(self):
        labels = [
            r"$\dot f$",
            r"$\dot\phi$",
            r"$\dot\theta$",
            r"$\dot\psi$",
        ]
        assert len(labels) == self.action_dim
        return labels
