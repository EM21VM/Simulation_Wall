import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib.Particle import Particle
from lib.Simulation import Boundary, Simulation
from lib.Wall import Wall
from lib.constants import npdarr

if __name__ == "__main__":
    print("Testing the Wall class")
    L = 500.0
    simulation = Simulation(
        dt=0.05,
        max_t=50.0,
        sides=[L, L, L],
        boundaries=[Boundary.WALL, Boundary.WALL, Boundary.EMPTY],
    )
    pos = np.array([0, 0, 0], dtype=np.float64)
    vel = np.array([1, 1, 0], dtype=np.float64)
    par = Particle(
        pos=pos,
        vel=vel,
        rad=5,
        mass=1,
        color="#00FF00",
        opacity=1,
    )
    simulation.add_object(par)
    wal = Wall(
        pos=np.array([L / 2, L / 2, 0]),
        end_pos=np.array([L / 2 + 100, L / 2 + 50, 0]),
        color="#FF0000",
        opacity=1,
    )
    simulation.add_object(wal)
    simulation.setup_system()
    simulation.put_particles_on_grid(
        Ns=np.array([10, 10, 1]), dL=np.array([20.0, 20.0, 20.0])
    )

    fig, ax = plt.subplots()
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    frame_label = ax.annotate(f"frame: 0/{simulation.num_steps:04d}", xy=(10, L - 10))
    circle = Circle(
        (par.pos[0], par.pos[1]), par.rad, facecolor=par.color, edgecolor="black", lw=1
    )
    ax.add_patch(circle)
    ax.add_patch(
        plt.Rectangle(
            (wal.pos[0], wal.pos[1]),
            width=wal.width,
            height=wal.height,
            facecolor=wal.color,
            alpha=wal.opacity,
            lw=1,
            angle=0
        )
    )
    plt.show()