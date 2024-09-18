import chex
import jax.numpy as jnp
import pytest

from phyjax2d import Space, SpaceBuilder


@pytest.fixture
def space() -> Space:
    builder = SpaceBuilder(gravity=(0.0, 0.0))

    builder.add_circle(radius=1.0)
    builder.add_circle(radius=2.0)
    builder.add_circle(radius=2.0)
    return builder.build()


def test_circle_to_circle(space: Space):
    sd = space.zeros_state()
    sd = sd.nested_ops("circle.p.xy", lambda xy: xy.at[0].set(jnp.array([1.0, 2.0])))
    sd = sd.nested_ops("circle.p.xy", lambda xy: xy.at[1].set(jnp.array([3.0, 2.0])))
    sd = sd.nested_ops("circle.p.xy", lambda xy: xy.at[2].set(jnp.array([4.0, 8.0])))
    contact = space.check_contacts_selected(sd, [("circle", "circle")])
    has_contact = contact.penetration >= 0
    chex.assert_trees_all_close(has_contact, jnp.array([True, False, False]))
