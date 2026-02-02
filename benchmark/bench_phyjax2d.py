import csv
from datetime import datetime, timedelta
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import typer

from phyjax2d import SpaceBuilder, Vec2d, step
from phyjax2d.moderngl_vis import MglVisualizer


def ball_fall_phyjax2d(
    n_balls: int,
    debug_vis: bool,
    n_iter: int = 1000,
) -> timedelta:
    """
    Simulates n_balls falling using phyjax2d.
    If debug_vis is True, uses MglVisualizer for rendering.
    """
    builder = SpaceBuilder(
        gravity=(0.0, -900.0),
        dt=0.002,
        jacobi_damping=0.5,
        n_velocity_iter=10,
        n_position_iter=2,
        bias_factor=0.02,
        bounce_threshold=4,
    )

    for _ in range(n_balls):
        builder.add_circle(
            radius=4.0,
            density=1.0 / (16 * np.pi),
            elasticity=0.5,
            friction=0.5,
        )

    # Container setup
    builder.add_segment(
        p1=Vec2d(50.0, 50.0),
        p2=Vec2d(550.0, 50.0),
        elasticity=0.4,
        friction=0.5,
    )
    builder.add_segment(
        p1=Vec2d(50.0, 50.0),
        p2=Vec2d(50.0, 800.0),
        elasticity=0.4,
        friction=0.5,
    )
    builder.add_segment(
        p1=Vec2d(550.0, 50.0),
        p2=Vec2d(550.0, 800.0),
        elasticity=0.4,
        friction=0.5,
    )

    space = builder.build()

    # 2. Initialize State
    rng = np.random.default_rng()
    x_coords = rng.uniform(70, 530, n_balls)
    y_coords = rng.uniform(400, 1000, n_balls)
    pos_array = jnp.stack([jnp.array(x_coords), jnp.array(y_coords)], axis=-1)

    sd = space.zeros_state().nested_replace("circle.p.xy", pos_array)
    vs = space.init_solver()

    # 3. Initialize Visualizer
    visualizer = None
    if debug_vis:
        # We define the range based on the window size/container
        visualizer = MglVisualizer(
            x_range=600.0,
            y_range=1000.0,
            space=space,
            stated=sd,
            title=f"Phyjax2D Debug: {n_balls} balls",
            figsize=(600, 1000),
        )

    # 4. Simulation Loop
    jit_step = jax.jit(step, static_argnums=(0,))

    # Warm-up (Optional: only if you want to exclude first-run JIT from benchmark)
    sd, _, _ = jit_step(space, sd, vs)

    start = datetime.now()
    for _ in range(n_iter):
        sd, _, _ = jit_step(space, sd, vs)

        if visualizer is not None:
            visualizer.render(state=sd)
            visualizer.show()

    duration = datetime.now() - start

    if visualizer is not None:
        visualizer.close()

    return duration


def main(
    count_min: int = 1000,
    count_max: int = 100000,
    count_interval: int = 1000,
    debug_vis: bool = False,
    filename: Path = Path("bench.csv"),
) -> None:
    results = []

    for count in [x for x in range(count_min, count_max + 1, count_interval)]:
        duration = ball_fall_phyjax2d(count, debug_vis)
        # Convert timedelta to total seconds as a float for the CSV
        seconds = duration.total_seconds()
        results.append((count, seconds))

    if not debug_vis:
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["n_balls", "duration_seconds"])  # Header
            writer.writerows(results)


if __name__ == "__main__":
    typer.run(main)
