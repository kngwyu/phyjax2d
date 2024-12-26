import chex
import jax.numpy as jnp
import pymunk
import pytest

from phyjax2d import Space, SpaceBuilder, Vec2d


@pytest.fixture
def space() -> Space:
    builder = SpaceBuilder(gravity=(0.0, 0.0))

    builder.add_circle(radius=2.0)
    builder.add_circle(radius=2.0)
    builder.add_polygon(points=[Vec2d(2.0, 0.0), Vec2d(0.0, 2.0), Vec2d(0.0, 0.0)])
    builder.add_polygon(
        points=[Vec2d(-1.0, 0.0), Vec2d(-3.0, 0.0), Vec2d(-2.0, -2.0)],
        radius=0.1,
    )
    return builder.build()


def test_triangle_to_circle(space: Space) -> None:
    sd = space.zeros_state().nested_replace(
        "circle.p.xy",
        jnp.array([[2.4, 2.4], [-2.0, -4.05]]),
    )
    contact = space.check_contacts_selected(sd, ("triangle", "circle"))
    has_contact = contact.penetration >= 0
    chex.assert_trees_all_close(has_contact, jnp.array([True, False, False, True]))
    sd = sd.nested_replace(
        "circle.p.xy",
        jnp.array([[2.42, 2.42], [-2.0, -4.11]]),
    )
    contact = space.check_contacts_selected(sd, ("triangle", "circle"))
    has_contact = contact.penetration >= 0
    chex.assert_trees_all_close(has_contact, jnp.array([False, False, False, False]))


def test_add_triangle() -> None:
    builder = SpaceBuilder(gravity=(0.0, 0.0))
    points = [Vec2d(2.0, 0.0), Vec2d(0.0, 2.0), Vec2d(-2.0, 0.0)]
    builder.add_polygon(points=points)
    polygon = builder.polygons[3][0]
    chex.assert_trees_all_close(
        polygon.points,
        jnp.array([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0]]),
    )
    chex.assert_trees_all_close(
        polygon.centroid,
        jnp.array([0.0, 2.0 / 3]),
    )
    root2 = 2**0.5
    chex.assert_trees_all_close(
        polygon.normals,
        jnp.array(
            [[1.0 / root2, 1.0 / root2], [-1.0 / root2, 1.0 / root2], [0.0, -1.0]]
        ),
    )
    chex.assert_trees_all_close(
        polygon.mass,
        jnp.array([4.0]),
    )
    chex.assert_trees_all_close(
        polygon.moment,
        jnp.array([pymunk.moment_for_poly(4.0, points)]),
    )


def test_add_square() -> None:
    builder = SpaceBuilder(gravity=(0.0, 0.0))
    builder.add_square(width=3.0, height=4.0)
    polygon = builder.polygons[4][0]
    chex.assert_trees_all_close(
        polygon.points,
        jnp.array([[1.5, 2.0], [-1.5, 2.0], [-1.5, -2.0], [1.5, -2.0]]),
    )
    chex.assert_trees_all_close(
        polygon.centroid,
        jnp.array([0.0, 0.0]),
    )
    chex.assert_trees_all_close(
        polygon.normals,
        jnp.array([[0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]]),
    )
    chex.assert_trees_all_close(
        polygon.mass,
        jnp.array([12.0]),
    )
    chex.assert_trees_all_close(
        polygon.moment,
        jnp.array([pymunk.moment_for_box(12.0, (3.0, 4.0))]),
    )


def test_add_hexagon() -> None:
    builder = SpaceBuilder(gravity=(0.0, 0.0))
    points = [
        Vec2d(4.0, 2.0),
        Vec2d(2.0, 4.0),
        Vec2d(-2.0, 4.0),
        Vec2d(-4.0, -2.0),
        Vec2d(-2.0, -4.0),
        Vec2d(2.0, -4.0),
    ]
    builder.add_polygon(points=points)
    polygon = builder.polygons[6][0]
    chex.assert_trees_all_close(
        polygon.centroid,
        jnp.array([0.0, 0.0]),
    )
    chex.assert_trees_all_close(
        polygon.mass,
        jnp.array([48.0]),
    )
    chex.assert_trees_all_close(
        polygon.moment,
        jnp.array([pymunk.moment_for_poly(48.0, points)]),
    )
