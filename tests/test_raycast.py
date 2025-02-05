import chex
import jax
import jax.numpy as jnp
import pytest

from phyjax2d import (
    Space,
    SpaceBuilder,
    Vec2d,
    circle_raycast,
    segment_raycast,
    thin_polygon_raycast,
)


@pytest.fixture
def space() -> Space:
    builder = SpaceBuilder(gravity=(0.0, 0.0))

    builder.add_circle(radius=2.0)
    builder.add_segment(p1=Vec2d(-2.0, -2.0), p2=Vec2d(2.0, 2.0))
    builder.add_polygon(points=[Vec2d(2.0, 0.0), Vec2d(0.0, 2.0), Vec2d(0.0, 0.0)])
    return builder.build()


def test_circle_raycast(space: Space) -> None:
    sd = space.zeros_state().nested_replace(
        "circle.p.xy",
        jnp.array([[0.0, 4.0]]),
    )
    p1 = jnp.array([[-4.0, 4.0], [-4.0, 3.0], [-4.0, 4.0]])
    p2 = jnp.array([[0.0, 4.0], [0.0, 3.0], [-8.0, 4.0]])
    rc = jax.vmap(circle_raycast, in_axes=(None, None, 0, 0, None, None))(
        0.0, 0.01, p1, p2, space.shaped.circle, sd.circle
    )
    chex.assert_trees_all_close(rc.fraction.ravel(), jnp.array([0.5, 0.5669873, -1.5]))


def test_segment_raycast(space: Space) -> None:
    sd = space.zeros_state()
    p1 = jnp.array([[-1.0, 1.0], [0.0, 2.0]])
    p2 = jnp.array([[1.0, -1.0], [1.0, 1.0]])
    rc = jax.vmap(segment_raycast, in_axes=(None, 0, 0, None, None))(
        0.01, p1, p2, space.shaped.segment, sd.segment
    )
    chex.assert_trees_all_close(rc.fraction.ravel(), jnp.array([0.5, 1.0]))


def test_polygon_raycast(space: Space) -> None:
    sd = space.zeros_state()
    p1 = jnp.array([[-2.0, 1.0], [0.0, -1.0]])
    p2 = jnp.array([[2.0, 1.0], [0.0, 9.0]])
    rc = jax.vmap(thin_polygon_raycast, in_axes=(None, 0, 0, None, None))(
        0.01, p1, p2, space.shaped.triangle, sd.triangle
    )
    chex.assert_trees_all_close(rc.fraction.ravel(), jnp.array([0.5, 0.1]))
