import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phyjax2d import Circle, Force, Position, Raycast, State, Velocity, XpbdSolver
from phyjax2d.tree_utils import compact_pytree_repr, pytree_repr


@compact_pytree_repr
class ExampleTuple(NamedTuple):
    array: object
    items: object


@compact_pytree_repr
@dataclasses.dataclass
class ExampleData:
    nested: object
    mapping: object
    sequence: object
    callback: object
    scalar: object


def test_namedtuple_repr() -> None:
    value = ExampleTuple(jnp.zeros((2, 3), dtype=jnp.float32), [1, 2])
    assert repr(value) == "ExampleTuple(\n  array=float32[2,3],\n  items=list[2]\n)"
    assert str(value) == repr(value)


@pytest.mark.parametrize(
    ("array", "expected"),
    [
        (np.zeros((), dtype=np.int32), "int32[]"),
        (np.zeros((0, 2), dtype=np.float64), "float64[0,2]"),
        (jnp.zeros((2, 1), dtype=bool), "bool[2,1]"),
    ],
)
def test_array_repr(array: object, expected: str) -> None:
    assert f"array={expected}," in repr(ExampleTuple(array, []))


def test_nested_containers_repr() -> None:
    value = ExampleData(ExampleTuple(None, []), {"key": 42}, (1, 2), len, None)
    assert repr(value) == (
        "ExampleData(\n  nested=ExampleTuple(...),\n  mapping=dict[1],"
        "\n  sequence=tuple[2],\n  callback=len,\n  scalar=None\n)"
    )
    assert str(value) == repr(value)


def test_unsupported_container() -> None:
    with pytest.raises(TypeError, match="is not a dataclass or NamedTuple"):
        pytree_repr(object())


@pytest.mark.parametrize("cls", [Position, Velocity, Force])
def test_position_like_repr(cls: type) -> None:
    value = cls(angle=jnp.zeros(3, dtype=jnp.float32), xy=jnp.ones((3, 2)))
    assert repr(value) == (
        f"{cls.__name__}(\n  angle=float32[3],\n  xy=float32[3,2]\n)"
    )
    assert str(value) == repr(value)
    updated = jax.jit(lambda tree: jax.tree.map(lambda x: x + 1, tree))(value)
    assert isinstance(updated, cls)
    np.testing.assert_array_equal(updated.angle, np.ones(3))
    np.testing.assert_array_equal(updated.xy, np.full((3, 2), 2))


def test_state_repr() -> None:
    value = State.zeros(2)
    assert repr(value) == (
        "State(\n  p=Position(...),\n  v=Velocity(...),\n  f=Force(...),"
        "\n  is_active=bool[2],\n  label=uint8[2]\n)"
    )
    assert str(value) == repr(value)


def test_shape_subclass_repr() -> None:
    value = Circle(
        mass=jnp.ones(2),
        moment=jnp.ones(2),
        elasticity=jnp.zeros(2),
        friction=jnp.zeros(2),
        rgba=jnp.zeros((2, 4), dtype=jnp.uint8),
        radius=jnp.ones(2),
    )
    assert repr(value) == (
        "Circle(\n  mass=float32[2],\n  moment=float32[2],"
        "\n  elasticity=float32[2],\n  friction=float32[2],"
        "\n  rgba=uint8[2,4],\n  radius=float32[2]\n)"
    )
    assert str(value) == repr(value)


def test_raycast_repr() -> None:
    value = Raycast(
        fraction=jnp.zeros(2),
        normal=jnp.zeros((2, 2)),
        hit=jnp.ones(2, dtype=bool),
    )
    assert repr(value) == (
        "Raycast(\n  fraction=float32[2],\n  normal=float32[2,2],\n  hit=bool[2]\n)"
    )
    assert str(value) == repr(value)


def test_solver_repr() -> None:
    value = XpbdSolver(
        lambda_n=jnp.zeros(0),
        lambda_t=jnp.zeros(0),
        contact=jnp.zeros(0, dtype=bool),
    )
    assert repr(value) == (
        "XpbdSolver(\n  lambda_n=float32[0],\n  lambda_t=float32[0],"
        "\n  contact=bool[0]\n)"
    )
    assert str(value) == repr(value)
