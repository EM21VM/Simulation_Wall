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
        dt=0.02,
        max_t=50.0,
        sides=[L, L, L],
        boundaries=[Boundary.WALL, Boundary.WALL, Boundary.WALL],
    )
    Particles = []
    Original_Pos = []
    for i in range(particle_num):
        # Creates Patricles within the Limits set and adds them to the Simulation
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
            for added_particle in Particles:
                distance = np.linalg.norm(pos - added_particle.pos)
                if distance < rad + added_particle.rad:
                    overlap = True
                    break
            if not overlap:
                Particles.append(Particle(pos=pos, vel=vel, rad=rad, mass=mass))
                # Saves the first Position the Particles have
                Original_Pos.append(pos)
                break
    simulation.add_objects(Particles)
    # Creates 4 Walls to make a little Box and adds to the Simulation
    Walls = [
        Wall(
            pos=np.array([100, 0, 100]),
            corner=np.array([100, 500, 200]),
            corner_2=np.array([100, 0, 400]),
            opacity=0.5,
            color="FF0000",
        ),
        Wall(
            pos=np.array([100, 0, 100]),
            corner=np.array([400, 0, 100]),
            corner_2=np.array([100, 500, 100]),
            opacity=0.5,
            color="66FF00",
        ),
        Wall(
            pos=np.array([400, 0, 100]),
            corner=np.array([400, 0, 400]),
            corner_2=np.array([400, 500, 100]),
            opacity=0.5,
        ),
        Wall(
            pos=np.array([100, 0, 400]),
            corner=np.array([400, 0, 400]),
            corner_2=np.array([100, 500, 400]),
            opacity=0.5,
            color="FFFF00",
        ),
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
        plotWall_pyv(
            wall, plotter, x_min=0, x_max=L, y_min=0, y_max=L, z_min=0, z_max=L
        )

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
