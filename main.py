import matplotlib.pyplot as plt
import numpy as np
from lib.Particle import Particle
from lib.Simulation import Boundary, Simulation
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

if __name__ == "__main__":
    print("Testing Simulation class")
    L = 500.0
    simulation = Simulation(
        dt=0.05,
        max_t=500.0,
        sides=[L, L, L],
        boundaries=[Boundary.PERIODIC, Boundary.WALL, Boundary.EMPTY],
    )

    p1 = Particle(
        pos=np.array([250, 250, 0]),
        vel=np.array([-15, 0, 0]),
        rad=10,
        mass=10,
        color="#00FFAA",
    )
    p2 = Particle(
        pos=np.array([400, 250, 0]),
        vel=np.array([10, 0, 0]),
        rad=5,
        mass=1,
        color="#FF0000",
    )
    simulation.add_object(p1)
    simulation.add_object(p2)
    simulation.setup_system()

    # Graphics
    fig, ax = plt.subplots()
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    frames_label = ax.annotate(
        f"frame: 0/{simulation.num_steps:04d}", xy=(10, L - 10)
    )
    circles = [
        Circle(
            pos[:2],
            particle.rad,
            facecolor=particle.color,
            edgecolor="black",
            lw=1,
        )
        for particle, pos in zip(
            simulation.particle_list, simulation.pos_matrix[0]
        )
    ]
    for circle in circles:
        ax.add_patch(circle)

    # Create animation
    def update_sphere_animation(frame):
        for pos, circle in zip(
            simulation.pos_matrix[frame],
            circles,
        ):
            circle.set_center(pos[:2])
        frames_label.set_text(f"frame: {frame:04d}/{simulation.num_steps:04d}")
        return [frames_label]

    # Run simulation and draw graphics
    simulation.run()
    animation = FuncAnimation(
        fig=fig,
        func=update_sphere_animation,
        frames=simulation.num_steps,
        interval=1,
    )
    plt.show()
