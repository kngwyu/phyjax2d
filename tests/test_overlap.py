import chex
import jax
import jax.numpy as jnp
import pytest

from phyjax2d import Space, SpaceBuilder, Vec2d, circle_overlap


@pytest.fixture
def space() -> Space:
    builder = SpaceBuilder(gravity=(0.0, 0.0))

    builder.add_circle(radius=2.0)
    builder.add_circle(radius=1.0)
    builder.add_segment(p1=Vec2d(-4.0, 0.0), p2=Vec2d(4.0, 0.0))
    builder.add_polygon(points=[Vec2d(2.0, 0.0), Vec2d(2.0, 2.0), Vec2d(0.0, 0.0)])
    builder.add_polygon(points=[Vec2d(2.0, 0.0), Vec2d(0.0, 2.0), Vec2d(0.0, 0.0)])
    return builder.build()


def test_circle_overlap(space: Space) -> None:
    sd = space.zeros_state().nested_replace(
        "circle.p.xy",
        jnp.array([[3.0, 3.0], [0.0, 0.0]]),
    )
    sd = sd.nested_replace("segment.p.xy", jnp.array([[0.0, -1.0]]))
    sd = sd.nested_replace("triangle.p.xy", jnp.array([[0.0, -5.0], [-2.0, -5.0]]))
    ol1 = circle_overlap(space.shaped, sd, jnp.array([0.0, 3.0]), 1.4)
    assert ol1.item()
    ol2 = circle_overlap(space.shaped, sd, jnp.array([0.0, 3.0]), 0.9)
    assert not ol2.item()
    ol3 = circle_overlap(space.shaped, sd, jnp.array([0.0, -4.0]), 1.0)
    assert ol3.item()
    ol3 = circle_overlap(space.shaped, sd, jnp.array([0.0, -4.0]), 1.0)
    assert ol3.item()
    ol4 = circle_overlap(space.shaped, sd, jnp.array([1.0, -6.0]), 1.1)
    assert ol4.item()
    ol5 = circle_overlap(space.shaped, sd, jnp.array([4.0, -5.0]), 1.8)
    assert not ol5.item()
    ol6 = circle_overlap(space.shaped, sd, jnp.array([-2.0, -3.0]), 0.1)
    assert ol6.item()
    ol7 = circle_overlap(space.shaped, sd, jnp.array([-2.0, 0.0]), 1.1)
    assert ol7.item()
