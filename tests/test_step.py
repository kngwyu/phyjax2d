import jax
import jax.numpy as jnp
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
