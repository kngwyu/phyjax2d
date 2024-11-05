import chex
import jax.numpy as jnp

from phyjax2d import SpaceBuilder, Vec2d


def test_add_triangle() -> None:
    builder = SpaceBuilder(gravity=(0.0, 0.0))
    builder.add_polygon(
        points=[Vec2d(2, 0), Vec2d(0.0, 2.0), Vec2d(-2, 0)],
    )
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


def test_add_hexagon() -> None:
    builder = SpaceBuilder(gravity=(0.0, 0.0))
    builder.add_polygon(
        points=[
            Vec2d(4.0, 2.0),
            Vec2d(2.0, 4.0),
            Vec2d(-2.0, 4.0),
            Vec2d(-4.0, -2.0),
            Vec2d(-2.0, -4.0),
            Vec2d(2.0, -4.0),
        ],
    )
    polygon = builder.polygons[6][0]
    chex.assert_trees_all_close(
        polygon.centroid,
        jnp.array([0.0, 0.0]),
    )
    chex.assert_trees_all_close(
        polygon.mass,
        jnp.array([48.0]),
    )
