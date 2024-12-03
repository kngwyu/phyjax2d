import jax
import jax.numpy as jnp
import typer

from phyjax2d import (
    Color,
    MglVisualizer,
    Space,
    SpaceBuilder,
    StateDict,
    Vec2d,
    VelocitySolver,
    make_square_segments,
)
from phyjax2d import step as phys_step

W, H = 600, 600

RED = Color(174, 68, 90)
PINK = Color(232, 188, 185)
NAVY = Color(75, 67, 118)
DARK_NAVY = Color(67, 46, 84)


def make_space() -> Space:
    builder = SpaceBuilder(gravity=(0.0, -9.8))
    walls = make_square_segments(0.0, 600.0, 0.0, 600.0)  # , rounded_offset=10.0
    builder.add_chain_segments(chain_points=walls)
    builder.add_circle(radius=40.0, color=RED)
    builder.add_circle(radius=40.0, color=RED)
    builder.add_circle(radius=60.0, color=PINK)
    builder.add_polygon(
        points=[Vec2d(40.0, 0.0), Vec2d(0.0, 40.0), Vec2d(0.0, 0.0)],
        color=NAVY,
    )
    builder.add_polygon(
        points=[Vec2d(30.0, 0.0), Vec2d(0.0, 30.0), Vec2d(-30.0, 0.0)],
        radius=0.1,
        color=DARK_NAVY,
    )
    return builder.build()


def main(n_steps: int = 100, scale: float = 1.0) -> None:
    space = make_space()
    sd = space.zeros_state().nested_replace(
        "circle.p.xy",
        jnp.array([[200.0, 300.0], [400.0, 300.0], [300.0, 400.0]]),
    )
    sd = sd.nested_replace(
        "triangle.p.xy",
        jnp.array([[150.0, 200.0], [500.0, 300.0]]),
    )
    sd = sd.nested_replace("triangle.p.angle", jnp.array([2.0, 1.0]))
    vis = MglVisualizer(
        x_range=W,
        y_range=H,
        space=space,
        stated=sd,
        figsize=(W * scale, H * scale),
    )
    vis.render(sd)
    vis.show()

    @jax.jit
    def step(sd: StateDict, solver: VelocitySolver) -> tuple[StateDict, VelocitySolver]:
        sd, solver, _ = phys_step(space, sd, solver)
        return sd, solver

    solver = space.init_solver()
    for _ in range(n_steps):
        sd, solver = step(sd, solver)
        vis.render(sd)
        vis.show()

    vis.close()


if __name__ == "__main__":
    app = typer.Typer(pretty_exceptions_show_locals=False)
    app.command()(main)
    app()
