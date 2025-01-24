import os
import sys
import numpy as np
import random
import pyvista as pv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib.Particle import Particle
from lib.Simulation import Boundary, Simulation
from lib.Wall import Wall, plotWall_pyv

if __name__ == "__main__":
    print("Testing the Wall class")
    # Sets the Size of the Plot and the max and min values of the Particles
    L = 500.0
    particle_num = 25
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
        dt=0.02,
        max_t=50.0,
        sides=[L, L, L],
        boundaries=[Boundary.WALL, Boundary.WALL, Boundary.WALL],
    )
    # Creates the Particles and saves their original Position for the Animation
    Particles = [
        Particle(np.array([150, 250, 250]), np.array([-50, 0, 0]), 10, 10),
        Particle(np.array([350, 250, 250]), np.array([50, 0, 0]), 10, 10),
    ]
    Original_Pos = []
    for particle in Particles:
        Original_Pos.append(particle.pos)

    simulation.add_objects(Particles)
    # Creates all the Walls and adds them to the Simulation
    Walls = [
        Wall(
            np.array([100, -100, 0]),
            np.array([100, 700, -200]),
            np.array([100, 0, 700]),
        ),
        Wall(np.array([400, 0, 0]), np.array([400, 500, 0]), np.array([400, 0, 500])),
    ]
    simulation.add_objects(Walls)
    # Setup the Simulation
    simulation.setup_system()

    # Initialising the Plotter and adding the box und Axes to it
    plotter = pv.Plotter()
    box = pv.Box(bounds=(0, L, 0, L, 0, L))
    plotter.add_mesh(box, color="blue", style="wireframe")
    plotter.add_axes(line_width=3, color="black")
    sphere_plots = []
    # Adds the Mesh of all Particles to a List
    for particle in simulation.particle_list:
        sphere = pv.Sphere(center=particle.pos, radius=particle.rad)
        actor = plotter.add_mesh(sphere, color=particle.color)
        sphere_plots.append(actor)
    # Adds the Mesh of all defined Walls
    for wall in simulation.wall_list:
        plotWall_pyv(wall, plotter, x_min=0, x_max=L, y_min=0, y_max=L, z_min=0, z_max=L)

    def update_plot(frame):
        global sphere_plots
        # Update the position of existing meshes
        for i, particle in enumerate(simulation.particle_list):
            pos = simulation.pos_matrix[frame][i]
            sphere_plots[i].SetPosition(pos - Original_Pos[i])
        plotter.update()
        plotter.render()

    # Run the Simulation and start the Animation
    simulation.run()
    plotter.show(interactive_update=True)
    for frame in range(simulation.num_steps):
        update_plot(frame)
