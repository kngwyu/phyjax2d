from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phyjax2d import Space, SpaceBuilder, Vec2d, step


@pytest.fixture
def space() -> Space:
    builder = SpaceBuilder(gravity=(0.0, -9.8))

    builder.add_circle(radius=2.0)
    builder.add_circle(radius=4.0)
    builder.add_segment(p1=Vec2d(-10.0, 0.0), p2=Vec2d(10.0, 0.0))
    return builder.build()


def test_circle_fall(space: Space) -> None:
    sd = space.zeros_state().nested_replace(
        "circle.p.xy",
        jnp.array([[-4.0, 6.0], [6.0, 8.0]]),
    )
    vs = space.init_solver()
    nsd, _, _ = step(space, sd, vs)
    assert nsd.circle.p.xy[0][1] < 6.0
    assert nsd.circle.p.xy[1][1] < 8.0


def test_circle_fall_jit(space: Space) -> None:
    sd = space.zeros_state().nested_replace(
        "circle.p.xy",
        jnp.array([[-4.0, 6.0], [6.0, 8.0]]),
    )
    vs = space.init_solver()
    jit_step = jax.jit(step, static_argnums=(0,))
    for _ in range(10):
        sd, _, _ = jit_step(space, sd, vs)
    assert sd.circle.p.xy[0][1] > 0.0
    assert sd.circle.p.xy[1][1] > 0.0


@pytest.mark.parametrize("n_circles", [2, 3])
@pytest.mark.parametrize("n_velocity_iter", [1, 6])
@pytest.mark.parametrize("use_jit", [False, True])
def test_warm_start_stops_approaching_circles(
    n_circles: int, n_velocity_iter: int, use_jit: bool
) -> None:
    builder = SpaceBuilder(
        gravity=(0.0, 0.0),
        linear_damping=0.0,
        angular_damping=0.0,
        bias_factor=0.0,
        bounce_threshold=100.0,
        n_position_iter=0,
        n_velocity_iter=n_velocity_iter,
    )
    for _ in range(n_circles):
        builder.add_circle(radius=1.0)
    space = builder.build()
    positions = jnp.stack((1.95 * jnp.arange(n_circles), jnp.zeros(n_circles)), axis=-1)
    velocities = jnp.zeros((n_circles, 2)).at[0, 0].set(1.0).at[-1, 0].set(-1.0)
    state = space.zeros_state().nested_replace("circle.p.xy", positions)
    state = state.nested_replace("circle.v.xy", velocities)
    contact = space.check_contacts(state)
    active = contact.penetration >= space.speculative_distance
    # Unit-radius, unit-density circles have mass pi. These cached impulses
    # exactly stop the end circles; both contributions to the middle cancel.
    expected_pn = jnp.where(active, jnp.pi, 0.0)
    solver = replace(space.init_solver(), contact=active, pn=expected_pn)
    step_fn = jax.jit(step, static_argnums=(0,)) if use_jit else step

    next_state, next_solver, _ = step_fn(space, state, solver)

    np.testing.assert_allclose(next_state.circle.v.xy, 0.0, atol=1e-6)
    np.testing.assert_allclose(next_state.circle.v.angle, 0.0, atol=1e-6)
    np.testing.assert_allclose(next_state.circle.p.xy, positions, atol=1e-6)
    np.testing.assert_allclose(next_solver.pn[active], expected_pn[active], atol=1e-6)
    np.testing.assert_allclose(next_solver.pt[active], 0.0, atol=1e-6)
