import numpy as np
import jax.numpy as jnp
import functools as ft
import jax
import jax.random as jr
from jax import Array

from scipy.linalg import inv, solve_discrete_are
from typing import Callable, Tuple, Any
from jax.lax import while_loop

from ..utils.typing import Array, Radius, BoolScalar, Pos, State, Action, PRNGKey
from ..utils.utils import merge01
from .obstacle import Obstacle, Rectangle, Cuboid, Sphere


def RK4_step(x_dot_fn: Callable, x: State, u: Action, dt: float) -> Array:
    k1 = x_dot_fn(x, u)
    k2 = x_dot_fn(x + 0.5 * dt * k1, u)
    k3 = x_dot_fn(x + 0.5 * dt * k2, u)
    k4 = x_dot_fn(x + dt * k3, u)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def lqr(
        A: np.ndarray,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray,
):
    """
    Solve the discrete time lqr controller.
        x_{t+1} = A x_t + B u_t
        cost = sum x.T*Q*x + u.T*R*u
    Code adapted from Mark Wilfred Mueller's continuous LQR code at
    https://www.mwm.im/lqr-controllers-with-python/
    Based on Bertsekas, p.151
    Yields the control law u = -K x
    """

    # first, try to solve the Riccati equation
    X = solve_discrete_are(A, B, Q, R)

    # compute the LQR gain
    K = inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    return K


def get_lidar(start_point: Pos, obstacles: Obstacle, num_beams: int, sense_range: float, max_returns: int = 32):
    if isinstance(obstacles, Rectangle):
        thetas = jnp.linspace(-np.pi, np.pi - 2 * np.pi / num_beams, num_beams)
        starts = start_point[None, :].repeat(num_beams, axis=0)
        ends = jnp.stack(
            [starts[..., 0] + jnp.cos(thetas) * sense_range, starts[..., 1] + jnp.sin(thetas) * sense_range],
            axis=-1)
    elif isinstance(obstacles, Cuboid) or isinstance(obstacles, Sphere):
        thetas = jnp.linspace(-np.pi / 2 + 2 * np.pi / num_beams, np.pi / 2 - 2 * np.pi / num_beams, num_beams // 2)
        phis = jnp.linspace(-np.pi, np.pi - 2 * np.pi / num_beams, num_beams)
        starts = start_point[None, :].repeat(thetas.shape[0] * phis.shape[0] + 2, axis=0)

        def get_end_point(theta, phi):
            return jnp.array([
                start_point[0] + jnp.cos(theta) * jnp.cos(phi) * sense_range,
                start_point[1] + jnp.cos(theta) * jnp.sin(phi) * sense_range,
                start_point[2] + jnp.sin(theta) * sense_range
            ])

        def get_end_point_theta(theta):
            return jax.vmap(lambda phi: get_end_point(theta, phi))(phis)

        ends = merge01(jax.vmap(get_end_point_theta)(thetas))
        ends = jnp.concatenate([ends,
                                start_point[None, :] + jnp.array([[0., 0., sense_range]]),
                                start_point[None, :] + jnp.array([[0., 0., -sense_range]])], axis=0)
    else:
        raise NotImplementedError
    sensor_data = raytracing(starts, ends, obstacles, max_returns)

    return sensor_data


def inside_obstacles(points: Pos, obstacles: Obstacle, r: Radius = 0.) -> BoolScalar:
    """
    points: (n, n_dim) or (n_dim, )
    obstacles: tree_stacked obstacles.

    Returns: (n, ) or (,). True if in collision, false otherwise.
    """
    # one point inside one obstacle
    def inside(point: Pos, obstacle: Obstacle):
        return obstacle.inside(point, r)

    # one point inside any obstacle
    def inside_any(point: Pos, obstacle: Obstacle):
        return jax.vmap(ft.partial(inside, point))(obstacle).max()

    # any point inside any obstacle
    if points.ndim == 1:
        if obstacles.center.shape[0] == 0:
            return jnp.zeros((), dtype=bool)
        is_in = inside_any(points, obstacles)
    else:
        if obstacles.center.shape[0] == 0:
            return jnp.zeros(points.shape[0], dtype=bool)
        is_in = jax.vmap(ft.partial(inside_any, obstacle=obstacles))(points)

    return is_in


def raytracing(starts: Pos, ends: Pos, obstacles: Obstacle, max_returns: int) -> Pos:
    # if the start point if inside the obstacle, return the start point
    is_in = inside_obstacles(starts, obstacles)

    def raytracing_single(start: Pos, end: Pos, obstacle: Obstacle):
        return obstacle.raytracing(start, end)

    def raytracing_any(start: Pos, end: Pos, obstacle: Obstacle):
        return jax.vmap(ft.partial(raytracing_single, start, end))(obstacle).min()

    if obstacles.center.shape[0] == 0:
        alphas = jnp.ones(starts.shape[0]) * 1e6
    else:
        alphas = jax.vmap(ft.partial(raytracing_any, obstacle=obstacles))(starts, ends)
        alphas *= (1 - is_in)

    # assert max_returns <= alphas.shape[0]
    alphas_return = jnp.argsort(alphas)[:max_returns]

    hitting_points = starts + (ends - starts) * (alphas[..., None])

    return hitting_points[alphas_return]



def get_node_goal_rng(
    key: PRNGKey,
    area_size: float,
    dim:int,
    obstacles : Obstacle,
    n_nodes:int,
    min_dist:float=None,
    max_travel:float =None,
    position_swap:bool=False,
    position_jitter:float =0.0,
    rotation_jitter:bool=False,
): #-> tuple[Array, Array] | tuple[Array, Array]:


    """
    Generate initial positions and goals.

    When position_swap=True, use the fixed 8-agent hardware
    configuration and assign opposite swap goals.
    """

    if position_swap:
        if n_nodes != 8:
            raise ValueError(
                "The fixed position-swap configuration requires 8 agents."
            )

        if dim != 2:
            raise ValueError(
                "The fixed position-swap configuration is defined in 2D."
            )

        # Coordinates centered at the workspace origin.
        # For area_size=2.0, these become positions inside [0, 2]^2.
        centered_xy = jnp.array(
            [
                [-1, -1],  # cf1
                [-1,  1],  # cf2
                [ 1,  1],  # cf3
                [ 1, -1],  # cf4
                [ 0.00, -0.707],  # cf5
                [-0.707,  0.00],  # cf6
                [ 0.00,  0.707],  # cf7
                [ 0.707,  0.00],  # cf8
            ],
            dtype=jnp.float32,
        )

        # Optional random global rotation during training.
        if rotation_jitter:
            key, rotation_key = jr.split(key)

            angle = jr.uniform(
                rotation_key,
                shape=(),
                minval=-jnp.pi,
                maxval=jnp.pi,
            )

            rotation_matrix = jnp.array(
                [
                    [jnp.cos(angle), -jnp.sin(angle)],
                    [jnp.sin(angle),  jnp.cos(angle)],
                ],
                dtype=jnp.float32,
            )

            centered_xy = centered_xy @ rotation_matrix.T

        # Optional independent perturbation of each initial position.
        if position_jitter > 0.0:
            key, jitter_key = jr.split(key)

            jitter = jr.uniform(
                jitter_key,
                shape=centered_xy.shape,
                minval=-position_jitter,
                maxval=position_jitter,
            )

            centered_xy = centered_xy + jitter

        # Shift coordinates from a centered frame into [0, area_size]^2.
        center = jnp.ones((2,), dtype=jnp.float32) * (
            area_size / 2.0
        )

        states = centered_xy + center

        # Keep agents away from the exact workspace boundary.
        boundary_margin = min_dist / 2.0

        states = jnp.clip(
            states,
            boundary_margin,
            area_size - boundary_margin,
        )

        # Swap pairs:
        # cf1 <-> cf3
        # cf2 <-> cf4
        # cf5 <-> cf7
        # cf6 <-> cf8
        swap_indices = jnp.array(
            [
                2,
                3,
                0,
                1,
                6,
                7,
                4,
                5,
            ],
            dtype=jnp.int32,
        )

        goals = states[swap_indices]

        return states, goals

def get_node_goal_rng_(
        key: PRNGKey,
        side_length: float,
        dim: int,
        obstacles: Obstacle,
        n: int,
        min_dist: float,
        max_travel: float = None,
        position_swap: bool = True,
        swap_radius: float = None,
) -> tuple[Array, Array] | tuple[Array, Array]:

    if position_swap:
        assert dim == 2, "Position-swap setup assumes 2D positions."
        assert n % 2 == 0, (
            "An even number of drones is required for opposite swapping."
        )

        if swap_radius is None:
            swap_radius = 0.35 * side_length

        # Training randomization parameters.
        radius_range = (0.90, 1.10)
        center_jitter = 0.03
        angle_jitter = 0.05
        position_jitter = 0.01

        (
            key_rotation,
            key_radius,
            key_center,
            key_angle,
            key_position,
        ) = jax.random.split(key, 5)

        # Randomly rotate the entire formation.
        global_rotation = jax.random.uniform(
            key_rotation,
            shape=(),
            minval=0.0,
            maxval=2.0 * jnp.pi,
        )

        # Randomize the circle radius.
        radius_scale = jax.random.uniform(
            key_radius,
            shape=(),
            minval=radius_range[0],
            maxval=radius_range[1],
        )
        radius = swap_radius * radius_scale

        # Keep the complete circle inside the workspace.
        workspace_margin = min_dist / 2.0 + position_jitter
        center_lower = radius + workspace_margin
        center_upper = side_length - radius - workspace_margin

        nominal_center = jnp.array(
            [side_length / 2.0, side_length / 2.0]
        )

        center_offset = jax.random.uniform(
            key_center,
            shape=(2,),
            minval=-center_jitter,
            maxval=center_jitter,
        )

        center = jnp.clip(
            nominal_center + center_offset,
            center_lower,
            center_upper,
        )

        n_half = n // 2
        nominal_angle_gap = 2.0 * jnp.pi / n

        # Account for the worst-case reduction in pairwise distance caused
        # by Cartesian position perturbations.
        required_chord_distance = (
                min_dist
                + 2.0 * jnp.sqrt(2.0) * position_jitter
        )

        required_angle_gap = 2.0 * jnp.arcsin(
            jnp.clip(
                required_chord_distance / (2.0 * radius),
                0.0,
                1.0 - 1e-6,
            )
        )

        # Two adjacent angular perturbations can reduce their angular gap by
        # at most 2 * allowed_angle_jitter.
        safe_angle_jitter = jnp.maximum(
            0.0,
            0.5 * (nominal_angle_gap - required_angle_gap),
        )

        allowed_angle_jitter = jnp.minimum(
            angle_jitter,
            safe_angle_jitter,
        )

        # Generate only one half of the circle.
        base_half_angles = (
                2.0
                * jnp.pi
                * jnp.arange(n_half)
                / n
        )

        half_angle_noise = jax.random.uniform(
            key_angle,
            shape=(n_half,),
            minval=-allowed_angle_jitter,
            maxval=allowed_angle_jitter,
        )

        first_half_theta = (
                base_half_angles
                + global_rotation
                + half_angle_noise
        )

        # Copy the first half after exactly pi radians. This guarantees that
        # agent i and agent i + n/2 remain diametrically opposite.
        theta = jnp.concatenate(
            [
                first_half_theta,
                first_half_theta + jnp.pi,
            ],
            axis=0,
        )

        states = center + radius * jnp.stack(
            [
                jnp.cos(theta),
                jnp.sin(theta),
            ],
            axis=1,
        )

        # Add small hardware initialization errors while preserving the
        # minimum-distance margin used above.
        states = states + jax.random.uniform(
            key_position,
            shape=(n, 2),
            minval=-position_jitter,
            maxval=position_jitter,
        )

        # Agent i targets the perturbed initial position of its opposite
        # agent i + n/2.
        goals = jnp.roll(
            states,
            shift=n_half,
            axis=0,
        )
        return states, goals


    max_iter = 1024  # maximum number of iterations to find a valid initial state/goal
    states = jnp.zeros((n, dim))
    goals = jnp.zeros((n, dim))

    def get_node(reset_input: Tuple[int, Array, Array, Array]):  # key, node, all nodes
        i_iter, this_key, _, all_nodes = reset_input
        use_key, this_key = jr.split(this_key, 2)
        i_iter += 1
        return i_iter, this_key, jr.uniform(use_key, (dim,), minval=0, maxval=side_length), all_nodes

    def non_valid_node(reset_input: Tuple[int, Array, Array, Array]):  # key, node, all nodes
        i_iter, _, node, all_nodes = reset_input
        dist_min = jnp.linalg.norm(all_nodes - node, axis=1).min()
        collide = dist_min <= min_dist
        inside = inside_obstacles(node, obstacles, r=min_dist)
        valid = ~(collide | inside) | (i_iter >= max_iter)
        return ~valid

    def get_goal(reset_input: Tuple[int, Array, Array, Array, Array]):
        # key, goal_candidate, agent_start_pos, all_goals
        i_iter, this_key, _, agent, all_goals = reset_input
        use_key, this_key = jr.split(this_key, 2)
        i_iter += 1
        if max_travel is None:
            return i_iter, this_key, jr.uniform(use_key, (dim,), minval=0, maxval=side_length), agent, all_goals
        else:
            return i_iter, this_key, jr.uniform(
                use_key, (dim,), minval=-max_travel, maxval=max_travel) + agent, agent, all_goals

    def non_valid_goal(reset_input: Tuple[int, Array, Array, Array, Array]):
        # key, goal_candidate, agent_start_pos, all_goals
        i_iter, _, goal, agent, all_goals = reset_input
        dist_min = jnp.linalg.norm(all_goals - goal, axis=1).min()
        collide = dist_min <= min_dist
        inside = inside_obstacles(goal, obstacles, r=min_dist)
        outside = jnp.any(goal < 0) | jnp.any(goal > side_length)
        if max_travel is None:
            too_long = np.array(False, dtype=bool)
        else:
            too_long = jnp.linalg.norm(goal - agent) > max_travel
        valid = (~collide & ~inside & ~outside & ~too_long) | (i_iter >= max_iter)
        out = ~valid
        assert out.shape == tuple() and out.dtype == jnp.bool_
        return out

    def reset_body(reset_input: Tuple[int, Array, Array, Array]):
        # agent_id, key, states, goals
        agent_id, this_key, all_states, all_goals = reset_input
        agent_key, goal_key, this_key = jr.split(this_key, 3)
        agent_candidate = jr.uniform(agent_key, (dim,), minval=0, maxval=side_length)
        n_iter_agent, _, agent_candidate, _ = while_loop(
            cond_fun=non_valid_node, body_fun=get_node,
            init_val=(0, agent_key, agent_candidate, all_states)
        )
        all_states = all_states.at[agent_id].set(agent_candidate)

        if max_travel is None:
            goal_candidate = jr.uniform(goal_key, (dim,), minval=0, maxval=side_length)
        else:
            goal_candidate = jr.uniform(goal_key, (dim,), minval=0, maxval=max_travel) + agent_candidate

        n_iter_goal, _, goal_candidate, _, _ = while_loop(
            cond_fun=non_valid_goal, body_fun=get_goal,
            init_val=(0, goal_key, goal_candidate, agent_candidate, all_goals)
        )
        all_goals = all_goals.at[agent_id].set(goal_candidate)
        agent_id += 1

        # if no solution is found, start over
        agent_id = (1 - (n_iter_agent >= max_iter)) * (1 - (n_iter_goal >= max_iter)) * agent_id
        all_states = (1 - (n_iter_agent >= max_iter)) * (1 - (n_iter_goal >= max_iter)) * all_states
        all_goals = (1 - (n_iter_agent >= max_iter)) * (1 - (n_iter_goal >= max_iter)) * all_goals

        return agent_id, this_key, all_states, all_goals

    def reset_not_terminate(reset_input: Tuple[int, Array, Array, Array]):
        # agent_id, key, states, goals
        agent_id, this_key, all_states, all_goals = reset_input
        return agent_id < n

    _, _, states, goals = while_loop(
        cond_fun=reset_not_terminate, body_fun=reset_body, init_val=(0, key, states, goals))


    return states, goals


def get_node_goal_rng_old(
        key: PRNGKey,
        side_length: float,
        dim: int,
        obstacles: Obstacle,
        n: int,
        min_dist: float,
        max_travel: float = None,
        position_swap: bool = False,
        swap_radius: float = None,
) -> tuple[Array, Array] | tuple[Array, Array]:

    # --------------------------------------------------------
    # Fixed circular position-swap setup
    # --------------------------------------------------------

    if position_swap:
        print("Position swap enabled")
        assert dim == 2, "Position swap setup assumes 2D positions."
        assert n % 2 == 0, "Need an even number of drones for opposite swapping."

        if swap_radius is None:
            swap_radius = 0.35 * side_length

        center = jnp.array([
            side_length / 2.0,
            side_length / 2.0,
        ])

        theta = 2.0 * jnp.pi * jnp.arange(n) / n

        states = center + swap_radius * jnp.stack(
            [
                jnp.cos(theta),
                jnp.sin(theta),
            ],
            axis=1,
        )

        # Drone i targets the initial position of drone i+n/2.
        goals = jnp.roll(states, shift=n // 2, axis=0)

        return states, goals


    # --------------------------------------------------------
    # Original random-reset code unchanged below
    # --------------------------------------------------------


    return states, goals


def get_node_goal_position_swap(
    side_length: float,
    n: int,
    radius: float,
    center: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create a circular initial configuration and opposite-position goals.

    Returns:
        states_xy: shape (n, 2), initial [x, y]
        goals_xy:  shape (n, 2), opposite [x_goal, y_goal]
    """
    assert n % 2 == 0, "Position swapping needs an even number of agents."

    if center is None:
        center = jnp.array([side_length / 2.0, side_length / 2.0])

    theta = 2.0 * jnp.pi * jnp.arange(n) / n

    states_xy = center + radius * jnp.stack(
        [jnp.cos(theta), jnp.sin(theta)],
        axis=1,
    )

    # Drone i goes to the initial position of drone i + n/2.
    goals_xy = jnp.roll(states_xy, shift=n // 2, axis=0)

    return states_xy, goals_xy
