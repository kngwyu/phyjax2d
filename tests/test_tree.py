import jax.numpy as jnp

from phyjax2d import Position


def test_split() -> None:
    pos = Position(angle=jnp.zeros((10, 1)), xy=jnp.ones((10, 2)))
    p1, p2 = pos.split(3)
    assert len(p1.angle) == 3
    assert len(p1.xy) == 3
    assert len(p2.angle) == 7
    assert len(p2.xy) == 7
