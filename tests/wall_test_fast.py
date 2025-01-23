import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import random
import pyvista as pv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib.Particle import Particle
from lib.Simulation import Boundary, Simulation
from lib.Wall import Wall, plotWall

if __name__ == "__main__":
    print("Testing the Wall class")
    L = 500.0
    particle_num = 1
    min_pos = 100
    max_pos = 400
    min_vel = 10
    max_vel = 100
    min_rad = 5
    max_rad = 10
    min_mass = 1
    max_mass = 20
    simulation = Simulation(
        dt=0.02,
        max_t=50.0,
        sides=[L, L, L],
        boundaries=[Boundary.WALL, Boundary.WALL, Boundary.WALL],
    )
    Particles = []
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
            for plotted_particle in Particles:
                distance = np.linalg.norm(pos - plotted_particle.pos)
                if distance < rad + plotted_particle.rad:
                    overlap = True
                    break
            if not overlap:
                Particles.append(Particle(pos=pos, vel=vel, rad=rad, mass=mass))
                break

    simulation.add_objects(Particles)
    Walls = [
        Wall(
            pos=np.array([100, 0, 0]),
            plains_vec=np.array([100, 500, 0]),
            plains_vec_2=np.array([100, 0, 500]),
            opacity= 0.5
        ),
        # Wall(
        #     pos=np.array([0, 0, 100]),
        #     plains_vec=np.array([500, 0, 100]),
        #     plains_vec_2=np.array([0, 500, 100]),
        # ),
        # Wall(
        #     pos=np.array([400, 0, 0]),
        #     plains_vec=np.array([400, 0, 500]),
        #     plains_vec_2=np.array([400, 500, 0]),
        # ),
        # Wall(
        #     pos=np.array([0, 0, 400]),
        #     plains_vec=np.array([500, 0, 400]),
        #     plains_vec_2=np.array([0, 500, 400]),
        # ),
    ]
    simulation.add_objects(Walls)

    simulation.setup_system()

    plotter = pv.Plotter()
    box = pv.Box(bounds=(0, L, 0, L, 0, L))
    plotter.add_mesh(box, color="blue", style="wireframe")
    plotter.add_axes(line_width=3, color="black")
    sphere_plots = []
    for particle in simulation.particle_list:
        sphere = pv.Sphere(center=particle.pos, radius=particle.rad)
        actor = plotter.add_mesh(sphere, color=particle.color)
        sphere_plots.append(actor)

    for wall in simulation.wall_list:
        if wall.normal_vec[2] != 0:  # Z-Ebene
            x = np.linspace(wall.min_cords[0], wall.max_cords[0], 100)
            y = np.linspace(wall.min_cords[1], wall.max_cords[1], 100)
            x, y = np.meshgrid(x, y)
            z = (
                -wall.distance_origin - wall.normal_vec[0] * x - wall.normal_vec[1] * y
            ) / wall.normal_vec[2]
            plotter.add_mesh(
                pv.StructuredGrid(x, y, z), color=wall.color, opacity=wall.opacity
            )
        elif wall.normal_vec[1] != 0:  # Y-Ebene
            x = np.linspace(wall.min_cords[0], wall.max_cords[0], 100)
            z = np.linspace(wall.min_cords[2], wall.max_cords[2], 100)
            x, z = np.meshgrid(x, z)
            y = (-wall.normal_vec[0] * x - wall.distance_origin) / wall.normal_vec[1]
            plotter.add_mesh(
                pv.StructuredGrid(x, y, z), color=wall.color, opacity=wall.opacity
            )
        elif wall.normal_vec[0] != 0:  # X-Ebene
            y = np.linspace(wall.min_cords[1], wall.max_cords[1], 100)
            z = np.linspace(wall.min_cords[2], wall.max_cords[2], 100)
            y, z = np.meshgrid(y, z)
            x = np.full_like(y, -wall.distance_origin / wall.normal_vec[0])
            plotter.add_mesh(
                pv.StructuredGrid(x, y, z), color=wall.color, opacity=wall.opacity
            )
    def update_plot(frame):
        global sphere_plots
        # Update the position of existing meshes
        for i, particle in enumerate(simulation.particle_list):
            pos = simulation.pos_matrix[frame][i]
            print("POS After: " + str(pos))
        sphere_plots[i].SetPosition(pos)  # Update position of the existing mesh
        plotter.render()
        plotter.update()

    simulation.run()
    plotter.show(interactive_update=True)
    for frame in range(simulation.num_steps):
        update_plot(frame)

    plotter.show()
