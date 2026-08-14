import functools as ft
import pathlib
from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from jax_f16.f16 import F16 as JaxF16, compute_f16_vel_angles
from jax_f16.f16_utils import rotx, roty, rotz

from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, AgentState, Array, Cost, Done, Info, Pos3d, Reward, State
from ..utils.utils import merge01, jax_vmap
from .base import MultiAgentEnv, RolloutResult
from .obstacle import Obstacle, Sphere
from .plot import render_video
from .utils import RK4_step, get_lidar, inside_obstacles, get_node_goal_rng


class F16Env(MultiAgentEnv):
    """
    Environment for multiple nonlinear F-16 aircraft.

    Native state:
      [VT, alpha, beta, phi, theta, psi, P, Q, R, PN, PE, H, POW, NZINT, PSINT, NYRINT]

    Outer control:
      [Nz_ref, ps_ref, Ny+r_ref, throttle_ref]

    All graph/safety positions use ENU = [PE, PN, H].
    Translational distances are in feet, consistent with jax-f16.
    """

    AGENT, GOAL, OBS = 0, 1, 2

    VT, ALPHA, BETA = JaxF16.VT, JaxF16.ALPHA, JaxF16.BETA
    PHI, THETA, PSI = JaxF16.PHI, JaxF16.THETA, JaxF16.PSI
    P, Q, R = JaxF16.P, JaxF16.Q, JaxF16.R
    PN, PE, H = JaxF16.PN, JaxF16.PE, JaxF16.H
    POW = JaxF16.POW
    NZINT, PSINT, NYRINT = JaxF16.NZINT, JaxF16.PSINT, JaxF16.NYRINT
    NZ, PS, NYR, THRTL = JaxF16.NZ, JaxF16.PS, JaxF16.NYR, JaxF16.THRTL

    class EnvState(NamedTuple):
        agent: AgentState
        goal: State
        obstacle: Obstacle

        @property
        def n_agent(self) -> int:
            return self.agent.shape[0]

    EnvGraphsTuple = GraphsTuple[State, EnvState]

    PARAMS = {
        "aircraft_radius": 25.0,
        "goal_radius": 150.0,
        "comm_radius": 3000.0,
        "n_rays": 16,
        "obs_len_range": [150.0, 600.0],
        "n_obs": 0,

        # Outer-loop command bounds.
        "nz_min": -1.0,
        "nz_max": 6.0,
        "ps_min": -1.0,
        "ps_max": 1.0,
        "nyr_min": -1.0,
        "nyr_max": 1.0,
        # Low-level equilibrium throttle is 0.1395 and the physical throttle
        # is clipped to [0,1], hence these outer-command bounds.
        "throttle_min": -0.1395,
        "throttle_max": 0.8605,

        # Numerical state envelope.
        "vt_min": 250.0,
        "vt_max": 800.0,
        "alpha_max": np.deg2rad(45.0),
        "beta_max": np.deg2rad(30.0),
        "theta_max": np.deg2rad(80.0),
        "rate_max": 4.0,
        "altitude_min": 0.0,
        "altitude_max": 60000.0,

        # Nominal waypoint controller.
        "max_bank_cmd": np.deg2rad(55.0),
        "max_pitch_cmd": np.deg2rad(25.0),
        "k_heading": 0.9,
        "k_roll": 1.5,
        "k_pitch": 2.0,
        "k_q": 0.6,
        "k_beta": 0.5,
        "k_r": 0.2,
        "k_speed": 0.002,
    }

    def __init__(self, num_agents: int, area_size: float, max_step: int = 256,
                 max_travel: float = None, dt: float = 0.03, params: dict = None):
        super().__init__(num_agents, area_size, max_step, max_travel, dt, params)
        self.create_obstacles = jax.vmap(Sphere.create)
        self.n_rays = min(16, self._params["n_rays"] ** 2 // 2 + 2)
        self._trim_state = jnp.asarray(JaxF16.trim_state(), dtype=jnp.float32)
        self._trim_control = jnp.asarray(JaxF16.trim_control(), dtype=jnp.float32)

    @property
    def state_dim(self) -> int:
        return JaxF16.NX

    @property
    def node_dim(self) -> int:
        return 3

    @property
    def edge_dim(self) -> int:
        # relative position, velocity, body-z axis, angular velocity
        return 12

    @property
    def action_dim(self) -> int:
        return JaxF16.NU

    @property
    def comm_radius(self):
        return self._params["comm_radius"]

    @property
    def trim_state(self):
        return self._trim_state

    @property
    def trim_control(self):
        return self._trim_control

    @staticmethod
    def _position_single(x: State) -> Pos3d:
        return jnp.array([x[JaxF16.PE], x[JaxF16.PN], x[JaxF16.H]])

    def position(self, x: State) -> Pos3d:
        if x.ndim == 1:
            return self._position_single(x)
        return jax.vmap(self._position_single)(x)

    @staticmethod
    def _wrap_angle(a):
        return jnp.arctan2(jnp.sin(a), jnp.cos(a))

    def reset(self, key: Array) -> GraphsTuple:
        self._t = 0
        n_obs = self._params["n_obs"]

        obs_key, key = jr.split(key)
        obs_pos = jr.uniform(obs_key, (n_obs, 3), minval=0.0, maxval=self.area_size)

        r_key, key = jr.split(key)
        obs_radius = jr.uniform(
            r_key, (n_obs,),
            minval=self._params["obs_len_range"][0] / 2,
            maxval=self._params["obs_len_range"][1] / 2,
        )
        obstacles = self.create_obstacles(obs_pos, obs_radius)

        positions, goal_positions = get_node_goal_rng(
            key, self.area_size, 3, obstacles, self.num_agents,
            4 * self._params["aircraft_radius"], self.max_travel
        )

        states = jnp.repeat(self.trim_state[None, :], self.num_agents, axis=0)
        goals = jnp.repeat(self.trim_state[None, :], self.num_agents, axis=0)

        states = states.at[:, self.PE].set(positions[:, 0])
        states = states.at[:, self.PN].set(positions[:, 1])
        states = states.at[:, self.H].set(positions[:, 2])
        goals = goals.at[:, self.PE].set(goal_positions[:, 0])
        goals = goals.at[:, self.PN].set(goal_positions[:, 1])
        goals = goals.at[:, self.H].set(goal_positions[:, 2])

        # psi=0 points North; atan2(East, North) gives the initial heading.
        dpos = goal_positions - positions
        psi0 = jnp.arctan2(dpos[:, 0], dpos[:, 1])
        states = states.at[:, self.PSI].set(psi0)
        goals = goals.at[:, self.PSI].set(psi0)

        return self.get_graph(self.EnvState(states, goals, obstacles))

    def step(self, graph: EnvGraphsTuple, action: Action, get_eval_info: bool = False):
        self._t += 1
        agent_states = graph.type_states(self.AGENT, self.num_agents)
        goal_states = graph.type_states(self.GOAL, self.num_agents)
        obstacles = graph.env_states.obstacle
        action = self.clip_action(action)

        next_agent_states = self.agent_step_rk4(agent_states, action)
        done = jnp.array(False)

        reward = -(jnp.linalg.norm(action - self.u_ref(graph), axis=1) ** 2).mean()
        cost = self.get_cost(graph)

        info = {}
        if get_eval_info:
            info["inside_obstacles"] = inside_obstacles(
                self.position(agent_states), obstacles, r=self._params["aircraft_radius"]
            )

        next_state = self.EnvState(next_agent_states, goal_states, obstacles)
        return self.get_graph(next_state), reward, cost, done, info

    def get_cost(self, graph: EnvGraphsTuple) -> Cost:
        x = graph.type_states(self.AGENT, self.num_agents)
        pos = self.position(x)
        dist = jnp.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        cost = (dist < 2 * self._params["aircraft_radius"]).any(axis=1).mean()
        cost += inside_obstacles(
            pos, graph.env_states.obstacle, r=self._params["aircraft_radius"]
        ).mean()
        return cost

    # ------------------------------------------------------------------
    # Nonlinear F-16 dynamics
    # ------------------------------------------------------------------

    @ft.partial(jax.jit, static_argnums=(0,))
    def _agent_xdot_single_agent(self, state: AgentState, control: Action) -> AgentState:
        control = self.clip_action(control)
        return JaxF16().xdot(state, control)

    def agent_xdot(self, states: AgentState, controls: Action) -> AgentState:
        if controls.ndim == 1:
            return self._agent_xdot_single_agent(states, controls)
        return jax_vmap(self._agent_xdot_single_agent)(states, controls)

    def agent_step_rk4(self, states: Array, controls: Array) -> Array:
        controls = self.clip_action(controls)
        x_new = RK4_step(self.agent_xdot, states, controls, self.dt)
        return self.clip_state(x_new)

    def rk4_single(self, state: Array, control: Array):
        control = self.clip_action(control)
        x_new = RK4_step(self._agent_xdot_single_agent, state, control, self.dt)
        return self.clip_state(x_new)

    # ------------------------------------------------------------------
    # Local control-affine approximation for GCBF+/CBF-QP baselines only
    # ------------------------------------------------------------------

    def control_affine_dyn_single(self, state: AgentState):
        u0 = self.trim_control
        xdot0 = self._agent_xdot_single_agent(state, u0)
        g = jax.jacobian(self._agent_xdot_single_agent, argnums=1)(state, u0)
        f = xdot0 - g @ u0
        return f, g

    def control_affine_dyn(self, state: State):
        assert state.ndim == 2
        return jax_vmap(self.control_affine_dyn_single)(state)

    # ------------------------------------------------------------------
    # Graph representation
    # ------------------------------------------------------------------

    def edge_state(self, state: State) -> State:
        def one(x):
            pos_enu = self._position_single(x)

            # jax-f16 velocity direction is in North-East-Up ordering.
            vel_neu = x[self.VT] * compute_f16_vel_angles(x)
            vel_enu = jnp.array([vel_neu[1], vel_neu[0], vel_neu[2]])

            R_neu = rotz(x[self.PSI]) @ roty(x[self.THETA]) @ rotx(x[self.PHI])

            z_neu = R_neu[:, 2]
            z_enu = jnp.array([z_neu[1], z_neu[0], z_neu[2]])

            omega_body = jnp.array([x[self.P], x[self.Q], x[self.R]])
            omega_neu = R_neu @ omega_body
            omega_enu = jnp.array([omega_neu[1], omega_neu[0], omega_neu[2]])

            return jnp.concatenate([pos_enu, vel_enu, z_enu, omega_enu])

        return jax.vmap(one)(state)

    def add_edge_feats(self, graph: GraphsTuple, state: State) -> GraphsTuple:
        edge_state = self.edge_state(state)
        edge_feats = edge_state[graph.receivers] - edge_state[graph.senders]

        norm = jnp.sqrt(1e-6 + jnp.sum(edge_feats[:, :3] ** 2, axis=-1, keepdims=True))
        safe_norm = jnp.maximum(norm, self.comm_radius)
        coef = jnp.where(norm > self.comm_radius, self.comm_radius / safe_norm, 1.0)
        edge_feats = edge_feats.at[:, :3].set(edge_feats[:, :3] * coef)
        return graph._replace(edges=edge_feats, states=state)

    def edge_blocks(self, state: EnvState, lidar_data: State) -> list[EdgeBlock]:
        n_hits = self.num_agents * self.n_rays
        agent_pos = self.position(state.agent)
        agent_edge_state = self.edge_state(state.agent)

        # agent-agent
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(self.num_agents) * (self.comm_radius + 1)
        state_diff = agent_edge_state[:, None, :] - agent_edge_state[None, :, :]
        aa_mask = dist < self.comm_radius
        id_agent = jnp.arange(self.num_agents)
        aa_edges = EdgeBlock(state_diff, aa_mask, id_agent, id_agent)

        # agent-goal
        id_goal = jnp.arange(self.num_agents, 2 * self.num_agents)
        ag_mask = jnp.eye(self.num_agents)
        goal_edge_state = self.edge_state(state.goal)
        ag_feats = agent_edge_state[:, None, :] - goal_edge_state[None, :, :]

        norm = jnp.sqrt(1e-6 + jnp.sum(ag_feats[..., :3] ** 2, axis=-1, keepdims=True))
        safe_norm = jnp.maximum(norm, self.comm_radius)
        coef = jnp.where(norm > self.comm_radius, self.comm_radius / safe_norm, 1.0)
        ag_feats = ag_feats.at[..., :3].set(ag_feats[..., :3] * coef)
        ag_edges = EdgeBlock(ag_feats, ag_mask, id_agent, id_goal)

        # agent-obstacle
        id_obs = jnp.arange(2 * self.num_agents, 2 * self.num_agents + n_hits)
        lidar_edge_state = self.edge_state(lidar_data)
        lidar_pos_all = self.position(lidar_data)
        ao_edges = []

        for i in range(self.num_agents):
            ids = jnp.arange(i * self.n_rays, (i + 1) * self.n_rays)
            lidar_pos = agent_pos[i] - lidar_pos_all[ids]
            lidar_feats = agent_edge_state[i] - lidar_edge_state[ids]
            lidar_dist = jnp.linalg.norm(lidar_pos, axis=-1)
            active = lidar_dist < self.comm_radius - 1e-1
            mask = jnp.logical_and(jnp.ones((1, self.n_rays)), active)
            ao_edges.append(
                EdgeBlock(lidar_feats[None, :, :], mask, id_agent[i][None], id_obs[ids])
            )

        return [aa_edges, ag_edges] + ao_edges

    def get_graph(self, state: EnvState) -> GraphsTuple:
        n_hits = self.n_rays * self.num_agents
        n_nodes = 2 * self.num_agents + n_hits

        node_feats = jnp.zeros((n_nodes, 3))
        node_feats = node_feats.at[:self.num_agents, 2].set(1)
        node_feats = node_feats.at[self.num_agents:2*self.num_agents, 1].set(1)
        node_feats = node_feats.at[-n_hits:, 0].set(1)

        node_type = jnp.zeros(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[self.num_agents:2*self.num_agents].set(self.GOAL)
        node_type = node_type.at[-n_hits:].set(self.OBS)

        get_lidar_vmap = jax.vmap(
            ft.partial(
                get_lidar,
                obstacles=state.obstacle,
                num_beams=self.params["n_rays"],
                sense_range=self.comm_radius,
                max_returns=self.n_rays,
            )
        )
        lidar_pos = merge01(get_lidar_vmap(self.position(state.agent)))

        # Zero native state for obstacle nodes, with only PE/PN/H populated.
        lidar_data = jnp.zeros((lidar_pos.shape[0], self.state_dim))
        lidar_data = lidar_data.at[:, self.PE].set(lidar_pos[:, 0])
        lidar_data = lidar_data.at[:, self.PN].set(lidar_pos[:, 1])
        lidar_data = lidar_data.at[:, self.H].set(lidar_pos[:, 2])

        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=self.edge_blocks(state, lidar_data),
            env_states=state,
            states=jnp.concatenate([state.agent, state.goal, lidar_data], axis=0),
        ).to_padded()

    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        agents = graph.type_states(self.AGENT, self.num_agents)
        goals = graph.type_states(self.GOAL, self.num_agents)
        obs = graph.type_states(self.OBS, self._params["n_rays"] * self.num_agents)

        action = self.clip_action(action)
        next_agents = self.agent_step_rk4(agents, action)
        next_states = jnp.concatenate([next_agents, goals, obs], axis=0)
        return self.add_edge_feats(graph, next_states)

    # ------------------------------------------------------------------
    # Nominal waypoint controller
    # ------------------------------------------------------------------

    def u_ref_inner_single(self, state: AgentState, goal: AgentState) -> Action:
        pos = self._position_single(state)
        goal_pos = self._position_single(goal)
        d = goal_pos - pos

        horizontal = jnp.sqrt(d[0] ** 2 + d[1] ** 2 + 1e-6)

        psi_des = jnp.arctan2(d[0], d[1])  # atan2(East, North)
        psi_err = self._wrap_angle(psi_des - state[self.PSI])

        phi_des = jnp.clip(
            self._params["k_heading"] * psi_err,
            -self._params["max_bank_cmd"],
            self._params["max_bank_cmd"],
        )
        ps_cmd = self._params["k_roll"] * (phi_des - state[self.PHI])

        theta_des = jnp.clip(
            jnp.arctan2(d[2], horizontal),
            -self._params["max_pitch_cmd"],
            self._params["max_pitch_cmd"],
        )
        nz_cmd = (
            self.trim_control[self.NZ]
            + self._params["k_pitch"] * (theta_des - state[self.THETA])
            - self._params["k_q"] * state[self.Q]
        )

        nyr_cmd = (
            self.trim_control[self.NYR]
            - self._params["k_beta"] * state[self.BETA]
            - self._params["k_r"] * state[self.R]
        )

        throttle_cmd = (
            self.trim_control[self.THRTL]
            + self._params["k_speed"] * (self.trim_state[self.VT] - state[self.VT])
        )

        return self.clip_action(jnp.array([nz_cmd, ps_cmd, nyr_cmd, throttle_cmd]))

    def u_ref(self, graph: GraphsTuple) -> Action:
        agents = graph.type_states(self.AGENT, self.num_agents)
        goals = graph.type_states(self.GOAL, self.num_agents)
        return jax_vmap(self.u_ref_inner_single)(agents, goals)

    # ------------------------------------------------------------------
    # Limits
    # ------------------------------------------------------------------

    def action_lim(self) -> Tuple[Action, Action]:
        low = jnp.array([
            self._params["nz_min"], self._params["ps_min"],
            self._params["nyr_min"], self._params["throttle_min"],
        ])
        high = jnp.array([
            self._params["nz_max"], self._params["ps_max"],
            self._params["nyr_max"], self._params["throttle_max"],
        ])
        return low, high

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        inf = jnp.inf
        low = jnp.array([
            self._params["vt_min"],
            -self._params["alpha_max"],
            -self._params["beta_max"],
            -inf,
            -self._params["theta_max"],
            -inf,
            -self._params["rate_max"],
            -self._params["rate_max"],
            -self._params["rate_max"],
            -inf, -inf,
            self._params["altitude_min"],
            0.0,
            -inf, -inf, -inf,
        ])
        high = jnp.array([
            self._params["vt_max"],
            self._params["alpha_max"],
            self._params["beta_max"],
            inf,
            self._params["theta_max"],
            inf,
            self._params["rate_max"],
            self._params["rate_max"],
            self._params["rate_max"],
            inf, inf,
            self._params["altitude_max"],
            100.0,
            inf, inf, inf,
        ])
        return low, high

    # ------------------------------------------------------------------
    # Safety masks
    # ------------------------------------------------------------------

    def safe_mask(self, graph: GraphsTuple) -> Array:
        pos = self.position(graph.type_states(self.AGENT, self.num_agents))
        dist = jnp.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        dist += jnp.eye(self.num_agents) * (2 * self._params["aircraft_radius"] + 1)
        safe_agent = jnp.min(dist > 4 * self._params["aircraft_radius"], axis=1)
        safe_obs = jnp.logical_not(
            inside_obstacles(pos, graph.env_states.obstacle, 2 * self._params["aircraft_radius"])
        )
        return jnp.logical_and(safe_agent, safe_obs)

    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        pos = self.position(graph.type_states(self.AGENT, self.num_agents))
        dist = jnp.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        dist += jnp.eye(self.num_agents) * (2 * self._params["aircraft_radius"] + 1)
        unsafe_agent = jnp.max(dist < 2.5 * self._params["aircraft_radius"], axis=1)
        unsafe_obs = inside_obstacles(
            pos, graph.env_states.obstacle, 1.5 * self._params["aircraft_radius"]
        )
        return jnp.logical_or(unsafe_agent, unsafe_obs)

    def collision_mask(self, graph: GraphsTuple) -> Array:
        pos = self.position(graph.type_states(self.AGENT, self.num_agents))
        dist = jnp.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        dist += jnp.eye(self.num_agents) * (2 * self._params["aircraft_radius"] + 1)
        unsafe_agent = jnp.max(dist < 2 * self._params["aircraft_radius"], axis=1)
        unsafe_obs = inside_obstacles(
            pos, graph.env_states.obstacle, self._params["aircraft_radius"]
        )
        return jnp.logical_or(unsafe_agent, unsafe_obs)

    def finish_mask(self, graph: GraphsTuple) -> Array:
        pos = self.position(graph.type_states(self.AGENT, self.num_agents))
        goal_pos = self.position(graph.env_states.goal)
        return jnp.linalg.norm(pos - goal_pos, axis=1) < self._params["goal_radius"]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_video(self, rollout: RolloutResult, video_path: pathlib.Path,
                     Ta_is_unsafe=None, viz_opts: dict = None, dpi: int = 100, **kwargs):
        # Standard GCBF+ renderer assumes positions are states[..., :3].
        # Make only a visualization copy with ENU positions in those slots.
        s = rollout.Tp1_graph.states
        rs = s.at[..., 0].set(s[..., self.PE])
        rs = rs.at[..., 1].set(s[..., self.PN])
        rs = rs.at[..., 2].set(s[..., self.H])
        rg = rollout.Tp1_graph._replace(states=rs)
        rr = rollout._replace(Tp1_graph=rg)

        render_video(
            rollout=rr, video_path=video_path,
            side_length=self.area_size, dim=3,
            n_agent=self.num_agents, n_rays=self.n_rays,
            r=self._params["aircraft_radius"],
            Ta_is_unsafe=Ta_is_unsafe, viz_opts=viz_opts, dpi=dpi, **kwargs
        )

    def render(self, graph: GraphsTuple) -> plt.Figure:
        pass

    @property
    def x_labels(self):
        return [
            "VT", "alpha", "beta", "phi", "theta", "psi",
            "P", "Q", "R", "PN", "PE", "H",
            "POW", "NZINT", "PSINT", "NYRINT"
        ]

    @property
    def uhl_labels(self):
        return ["Nz_ref", "ps_ref", "Ny+r_ref", "throttle_ref"]
