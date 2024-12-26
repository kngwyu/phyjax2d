"""Raycast"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import chex

from phyjax2d.impl import (
    Circle,
    Segment,
    State,
    normalize,
    _vmap_dot,
    _sv_cross,
)


@chex.dataclass
class Raycast:
    fraction: jax.Array
    normal: jax.Array
    hit: jax.Array


def circle_raycast(
    radius: float | jax.Array,
    max_fraction: float | jax.Array,
    p1: jax.Array,
    p2: jax.Array,
    circle: Circle,
    state: State,
) -> Raycast:
    # Suppose p1 and p2's shape has (2,)
    s = jnp.expand_dims(p1, axis=0) - state.p.xy  # (N, 2)
    d, length = normalize(p2 - p1)
    t = -jnp.dot(s, d)  # (N,)

    @jax.vmap
    def muld(x: jax.Array) -> jax.Array:
        return x * d

    c = s + muld(t)  # (N, 2)
    cc = _vmap_dot(c, c)  # (N, 1)
    rr = (radius + circle.radius) ** 2
    fraction = jnp.where(rr >= cc, t - jnp.sqrt(rr - cc), 0.0)
    hitpoint = s + muld(fraction)
    normal, _ = normalize(hitpoint)
    return Raycast(  # type: ignore
        fraction=fraction / length,
        normal=normal,
        hit=jnp.logical_and(
            cc <= rr,
            jnp.logical_and(0.0 <= fraction, fraction <= max_fraction * length),
        ),
    )


def segment_raycast(
    max_fraction: float | jax.Array,
    p1: jax.Array,
    p2: jax.Array,
    segment: Segment,
    state: State,
) -> Raycast:
    d = p2 - p1
    v1, v2 = segment.point1, segment.point2
    v1, v2 = state.p.transform(v1), state.p.transform(v2)
    e = v2 - v1
    eunit, length = normalize(e)
    normal = _sv_cross(jnp.ones_like(length) * -1, eunit)
    numerator = _vmap_dot(normal, v1 - p1)  # (N,)
    denominator = jnp.dot(normal, d)  # (N,)
    t = numerator / denominator
    p = jax.vmap(lambda ti: ti * d + p1)(t)  # (N, 2)
    s = _vmap_dot(p - v1, eunit)
    normal = jnp.where(jnp.expand_dims(numerator > 0.0, axis=1), -normal, normal)
    return Raycast(  # type: ignore
        fraction=t,
        normal=normal,
        hit=jnp.logical_and(
            denominator != 0.0,
            jnp.logical_and(
                jnp.logical_and(t >= 0.0, t <= max_fraction),
                jnp.logical_and(s >= 0.0, s <= length),
            ),
        ),
    )
