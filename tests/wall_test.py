import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib.Particle import Particle
from lib.Simulation import Boundary, Simulation
from lib.Wall import Wall, plotWall
from lib.constants import npdarr, Axes

if __name__ == "__main__":
    print("Testing the Wall class")
    L = 500.0
    simulation = Simulation(
        dt=0.05,
        max_t=50.0,
        sides=[L, L, L],
        boundaries=[Boundary.WALL, Boundary.WALL, Boundary.WALL],
    )
    pos = np.array([L / 2, L / 2, L / 2])
    vel = np.array([0, 100, 0])
    rad = 10
    par = Particle(
        pos=pos + rad,
        vel=vel,
        rad=rad,
        mass=1,
        color="#00FF00",
        opacity=1,
    )
    simulation.add_object(par)
    wal = Wall(
        pos=np.array([500, 100, 500]),
        plains_vec=np.array([500, 100, 0]),
        plains_vec_2=np.array([0, 100, 0]),
        offset=50,
    )
    simulation.add_object(wal)
    simulation.setup_system()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_zlim(0, L)
    x_min, x_max = 0, L
    y_min, y_max = 0, L
    z_min, z_max = 0, L
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    ax.set_aspect("equal")
    frames_label = ax.annotate(f"frame: 0/{simulation.num_steps:04d}", xy=(10, L - 10))
    overlaps_label = ax.annotate("overlaps: []", (20, 20))
    plotWall(wal, ax, x_min, x_max, y_min, y_max, z_min, z_max)

    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)
    x0 = par.pos[0] + par.rad * np.sin(phi) * np.cos(theta)
    y0 = par.pos[1] + par.rad * np.sin(phi) * np.sin(theta)
    z0 = par.pos[2] + par.rad * np.cos(phi)

    sphere_plots= []
    def init():
        global sphere_plots
        sphere_plots.clear()
        for particle in simulation.particle_list:
            pos = particle.pos
            sphere_plot = ax.plot_surface(
                pos[0] + particle.rad * np.sin(phi) * np.cos(theta),
                pos[1] + particle.rad * np.sin(phi) * np.sin(theta),
                pos[2] + particle.rad * np.cos(phi),
                color=particle.color,
            )
            sphere_plots.append(sphere_plot)
        return sphere_plots

    def update_sphere_animation(frame):
        global sphere_plots
        for i, particle in enumerate(simulation.particle_list):
            pos = simulation.pos_matrix[frame][i]
            print(str(pos))
            sphere_plots[i].remove()
            sphere_plots[i] = ax.plot_surface(
                pos[0] + particle.rad * np.sin(phi) * np.cos(theta),
                pos[1] + particle.rad * np.sin(phi) * np.sin(theta),
                pos[2] + particle.rad * np.cos(phi),
                color=particle.color,
            )
            frames_label.set_text(f"frame: {frame}/{simulation.num_steps:04d}")
        return sphere_plots

    simulation.run()
    animation = FuncAnimation(
        fig=fig,
        func=update_sphere_animation,
        frames=simulation.num_steps,
        interval=1,
        init_func=init,
    )
    plt.show()
