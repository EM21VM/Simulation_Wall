import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib.Particle import Particle
from lib.Simulation import Boundary, Simulation
from lib.Wall import Wall, plotWall

if __name__ == "__main__":
    print("Testing the Wall class")
    # Sets the Size of the Plot and the max and min values of the Particles
    L = 500.0
    particle_num = 5
    min_pos = 100
    max_pos = 400
    min_vel = 10
    max_vel = 100
    min_rad = 5
    max_rad = 10
    min_mass = 1
    max_mass = 20
    # Sets up the simulation
    simulation = Simulation(
        dt=0.05,
        max_t=50.0,
        sides=[L, L, L],
        boundaries=[Boundary.WALL, Boundary.WALL, Boundary.WALL],
    )
    Particles = []
    # Creates Patricles within the Limits set and adds them to the Simulation
    for i in range(particle_num):
        while True:
            rad = random.randint(min_rad, max_rad)
            mass = random.randint(min_mass, max_mass)
            pos = np.array(
                [
                    random.randint(min_pos + rad, max_pos - rad),
                    random.randint(min_pos + rad, max_pos - rad),
                    random.randint(min_pos + rad, max_pos - rad),
                ]
            )
            vel = np.array(
                [
                    random.randint(min_vel, max_vel),
                    random.randint(min_vel, max_vel),
                    random.randint(min_vel, max_vel),
                ]
            )
            overlap = False
            # Checks the Distance to all created Particles and if the Particle is too close create a new one and check again
            for plotted_particle in Particles:
                distance = np.linalg.norm(pos - plotted_particle.pos)
                if distance < rad + plotted_particle.rad:
                    overlap = True
                    break
            if not overlap:
                Particles.append(Particle(pos=pos, vel=vel, rad=rad, mass=mass))
                break

    simulation.add_objects(Particles)
    # Creates 4 Walls to make a little Box and adds to the Simulation
    Walls = [
        Wall(
            pos=np.array([100, 0, 0]),
            plains_vec=np.array([100, 500, 0]),
            plains_vec_2=np.array([100, 0, 500]),
        ),
        Wall(
            pos=np.array([0, 0, 100]),
            plains_vec=np.array([500, -100, 100]),
            plains_vec_2=np.array([-100, 500, 100]),
        ),
        Wall(
            pos=np.array([400, 0, 0]),
            plains_vec=np.array([400, -100, 500]),
            plains_vec_2=np.array([400, 500, -100]),
        ),
        Wall(
            pos=np.array([0, 0, 400]),
            plains_vec=np.array([700, -200, 400]),
            plains_vec_2=np.array([-200, 700, 400]),
        ),
    ]
    simulation.add_objects(Walls)

    # Setup the Simulation and define the Attributes of the Animation
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
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_aspect("equal")
    frames_label = ax.annotate(f"frame: 0/{simulation.num_steps:04d}", xy=(10, L - 10))
    overlaps_label = ax.annotate("overlaps: []", (20, 20))
    # Plots all walls in the simulation
    for wall in simulation.wall_list:
        plotWall(wall, ax, x_min, x_max, y_min, y_max, z_min, z_max)

    # Sets up the values to plot a sphere 
    theta = np.linspace(0, 2 * np.pi, 50)
    phi = np.linspace(0, np.pi, 50)
    theta, phi = np.meshgrid(theta, phi)
    sphere_plots = []

    def init():
        # Plots all the spheres in the Simulation on their starting position
        global sphere_plots
        sphere_plots.clear()
        for particle in simulation.particle_list:
            pos = particle.pos
            sphere_plot = ax.plot_surface(
                pos[0] + particle.rad * np.sin(phi) * np.cos(theta),
                pos[1] + particle.rad * np.sin(phi) * np.sin(theta),
                pos[2] + particle.rad * np.cos(phi),
                color=particle.color,
                alpha=particle.opacity,
            )
            sphere_plots.append(sphere_plot)
        return sphere_plots

    def update_sphere_animation(frame):
        global sphere_plots
        # Delete the Plots of the spheres from the previous frame and Plot the spheres on the new Posistion 
        for i, particle in enumerate(simulation.particle_list):
            pos = simulation.pos_matrix[frame][i]
            sphere_plots[i].remove()
            sphere_plots[i] = ax.plot_surface(
                pos[0] + particle.rad * np.sin(phi) * np.cos(theta),
                pos[1] + particle.rad * np.sin(phi) * np.sin(theta),
                pos[2] + particle.rad * np.cos(phi),
                color=particle.color,
                alpha=particle.opacity,
            )
        return sphere_plots

    # Run the Simulation and the Plot
    simulation.run()
    animation = FuncAnimation(
        fig=fig,
        func=update_sphere_animation,
        frames=simulation.num_steps,
        interval=10,
        init_func=init,
    )
    plt.show()
