import jax.numpy as jnp
import pytest

from phyjax2d import Space, SpaceBuilder
from phyjax2d.impl import _circle_to_circle_impl


@pytest.fixture
def space() -> Space:
    builder = SpaceBuilder(gravity=(0.0, 0.0))

    builder.add_circle(radius=1.0)
    builder.add_circle(radius=2.0)
    builder.add_circle(radius=2.0)
    builder.add_circle(radius=4.0)
    return builder.build()


def test_circle_to_circle(space: Space):
    sd = space.zeros_state()
    sd = sd.nested_ops("circle.p.xy", lambda xy: xy.at[0].set(jnp.array([1.0, 2.0])))
    sd = sd.nested_ops("circle.p.xy", lambda xy: xy.at[1].set(jnp.array([4.0, 2.0])))
    contact = _circle_to_circle_impl(
        space.shaped.circle,
        space.shaped.circle,
        sd.circle.p,
        sd.circle.p,
        jnp.ones(4, dtype=bool),
    )
