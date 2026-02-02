import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
import typer


def ball_fall(n_balls: int, debug_vis: bool, n_iter: int = 1000) -> timedelta:
    space = pymunk.Space()
    # 1. Flip Gravity: Positive Y pulls "down" in PyGame coordinates
    space.gravity = (0, 900)

    static_body = space.static_body
    # 2. Invert Container: Floor is now at Y=800, walls go up toward Y=50
    segments = [
        pymunk.Segment(static_body, (50, 800), (550, 800), 5),  # Floor (Bottom)
        pymunk.Segment(static_body, (50, 800), (50, 50), 5),  # Left Wall
        pymunk.Segment(static_body, (550, 800), (550, 50), 5),  # Right Wall
    ]
    for seg in segments:
        seg.elasticity = 0.4
        seg.friction = 0.5
    space.add(*segments)

    radius = 4
    mass = 1
    moment = pymunk.moment_for_circle(mass, 0, radius)

    rng = np.random.default_rng()
    # 3. Invert Spawn: Balls start near the top (Y=100 to 400)
    x_coords = rng.uniform(70, 530, n_balls)
    y_coords = rng.uniform(50, 400, n_balls)

    for i in range(n_balls):
        body = pymunk.Body(mass, moment)
        body.position = (float(x_coords[i]), float(y_coords[i]))
        shape = pymunk.Circle(body, radius)
        shape.elasticity = 0.5
        shape.friction = 0.5
        space.add(body, shape)

    if debug_vis:
        pygame.init()
        screen = pygame.display.set_mode((600, 850))
        pygame.display.set_caption(f"Inverted Gravity Benchmark: {n_balls} balls")
        draw_options = pymunk.pygame_util.DrawOptions(screen)
    else:
        screen = None
        draw_options = None

    start = datetime.now()
    for _ in range(n_iter):
        if screen is not None and draw_options is not None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return datetime.now() - start

            screen.fill((255, 255, 255))
            space.debug_draw(draw_options)
            pygame.display.flip()

        space.step(0.002)

    if debug_vis:
        pygame.quit()

    return datetime.now() - start


DEFAULT_COUNTS = [1000]


def main(
    counts: list[int] = DEFAULT_COUNTS,
    debug_vis: bool = False,
    filename: Path = Path("bench.csv"),
) -> None:
    results = []

    for count in counts:
        duration = ball_fall(count, debug_vis)
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
