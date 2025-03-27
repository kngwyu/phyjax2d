import chex
import jax.numpy as jnp
import pytest

from phyjax2d import Space, SpaceBuilder, Vec2d, step


@pytest.fixture
def space() -> Space:
    builder = SpaceBuilder(gravity=(0.0, 0.0))

    builder.add_circle(radius=1.0)
    builder.add_circle(radius=2.0)
    builder.add_circle(radius=2.0)
    builder.add_segment(p1=Vec2d(-2.0, 0.0), p2=Vec2d(2.0, 0.0))
    builder.add_segment(p1=Vec2d(0.0, -2.0), p2=Vec2d(0.0, 2.0))
    builder.add_polygon(points=[Vec2d(2.0, 0.0), Vec2d(0.0, 2.0), Vec2d(0.0, 0.0)])
    builder.add_polygon(points=[Vec2d(2.0, 0.0), Vec2d(2.0, 2.0), Vec2d(0.0, 0.0)])
    return builder.build()


def test_circle_to_circle(space: Space) -> None:
    sd = space.zeros_state().nested_replace(
        "circle.p.xy",
        jnp.array([[1.0, 2.0], [3.0, 2.0], [4.0, 8.0]]),
    )
    contact = space.check_contacts_selected(sd, ("circle", "circle"))
    has_contact = contact.penetration >= 0
    chex.assert_trees_all_close(has_contact, jnp.array([True, False, False]))


def test_circle_to_segment(space: Space) -> None:
    sd = space.zeros_state().nested_replace(
        "circle.p.xy",
        jnp.array([[1.0, 2.0], [3.0, 2.0], [4.0, 8.0]]),
    )
    sd = sd.nested_replace("segment.p.xy", jnp.array([[0.0, 2.0], [4.0, 4.0]]))
    contact = space.check_contacts_selected(sd, ("segment", "circle"))
    has_contact = contact.penetration >= 0
    chex.assert_trees_all_close(
        has_contact,
        jnp.array([True, True, False, False, True, True]),
    )
    chex.assert_trees_all_close(
        contact.pos[:2],
        jnp.array([[1.0, 2.0], [1.5, 2.0]]),
    )
    chex.assert_trees_all_close(
        contact.pos[-2:],
        jnp.array([[4.5, 2.0], [4.0, 6.0]]),
    )


def test_circle_to_triangle(space: Space) -> None:
    sd = space.zeros_state().nested_replace(
        "circle.p.xy",
        jnp.array([[1.0, 2.0], [3.0, 2.0], [4.0, 5.9]]),
    )
    sd = sd.nested_replace("triangle.p.xy", jnp.array([[0.0, 3.5], [4.0, 4.0]]))
    contact = space.check_contacts_selected(sd, ("triangle", "circle"))
    has_contact = contact.penetration >= 0
    chex.assert_trees_all_close(
        has_contact,
        jnp.array([False, True, False, False, False, True]),
    )


def test_ignore() -> None:
    builder = SpaceBuilder(gravity=(0.0, 0.0))

    builder.add_circle(radius=1.0)
    builder.add_circle(radius=2.0, ignore=["circle"])
    builder.add_circle(radius=2.0, ignore=["static_circle"])
    builder.add_circle(radius=3.0, is_static=True)
    builder.add_circle(radius=3.0, is_static=True)
    space = builder.build()
    sd = space.zeros_state().nested_replace(
        "circle.p.xy",
        jnp.array([[1.0, 2.0], [3.0, 2.0], [4.0, 8.0]]),
    )
    sd = sd.nested_replace("static_circle.p.xy", jnp.array([[0.0, 2.0], [8.0, 8.0]]))
    contact = space.check_contacts_selected(sd, ("circle", "circle"))
    has_contact = contact.penetration >= 0
    chex.assert_trees_all_close(has_contact, jnp.array([False]))
    contact = space.check_contacts_selected(sd, ("circle", "static_circle"))
    has_contact = contact.penetration >= 0
    chex.assert_trees_all_close(has_contact, jnp.array([True, False, True, False]))

    solver = space.init_solver()
    step(space, sd, solver)
