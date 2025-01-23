import numpy as np
import sys
import os
import time as time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib.Particle import Particle
from lib.constants import ZERO_VEC, npdarr, Axes
from lib.Object import Object


class Wall(Object):
    """A Wall in a 3-Dimensinal Space which allows particles to bounce of it


    Attributes:
        pos (npdarr): The first point of the Wall (required)
        plains_vec (npdarr): The second point of the Wall (required)
        plains_vec2 (npdarr): The third point of the Wall (required)
        color (string): The color of the Wall (default: "#FF0000")
        opacity (float): The opacity of the Wall (default: 1.0)
    """

    def __init__(
        # Formula for plains
        # a*x + b*y + c*z + d = 0
        # normal_vec[0] * x + normal_vec[1] * y + normal_vec[2] * z + distance_origin = 0
        self,
        # Defaults of the Wall Class
        pos: npdarr = np.copy(ZERO_VEC),
        plains_vec: npdarr = np.copy(ZERO_VEC),
        plains_vec_2: npdarr = np.copy(ZERO_VEC),
        color: str = "#0000FF",
        opacity: float = 1.0,
    ) -> None:
        # Initialize the Object with the given Attributes
        self.pos = pos
        self.plains_vec = plains_vec
        self.plains_vec_2 = plains_vec_2
        self.color = color
        self.opacity = opacity

        # Checking if all required Attributes are set
        if np.all(plains_vec == 0) or np.all(plains_vec_2 == 0):
            raise ValueError(
                'The Wall needs to have 3 Points as Arguments and only Attribute "pos" can be the zero vector'
            )

            # Calculates the Normal Vector and normalizes it

        self.normal_vec = np.cross(pos - plains_vec, pos - plains_vec_2)
        self.normal_vec = self.normal_vec / np.linalg.norm(self.normal_vec)

        # Calculates the Distance from Origin (d in the formular for plains)
        self.distance_origin = -pos.dot(self.normal_vec)
        # Gets the LLF and RHB Points from the given Points
        bbox_pts = get_bbox_pts(self)
        self.min_cords = bbox_pts[0]
        self.max_cords = bbox_pts[1]

        print("POS: " + str(pos))
        print("P1: " + str(plains_vec) + " P2: " + str(plains_vec_2))
        print("AB: " + str(pos - plains_vec) + "AC: " + str(pos - plains_vec_2))
        print("Normalvector: " + str(self.normal_vec))
        print("Distance to origin: " + str(self.distance_origin))
        print("BBox: " + str(bbox_pts))

        # Initialzies the super class Object with given Attributes
        super().__init__(pos, bbox_pts=bbox_pts, color=color, opacity=opacity)


def wall_collision(particle: Particle, wall: Wall) -> npdarr:
    """
    Returns the new Velocity Vector the Particle should have when hitting the Wall

    Parameters:
     particle (Particle): The Particle colliding with the Wall
     wall (Wall): The Wall the Particle collides with
    """
    v = particle.vel
    n = wall.normal_vec
    if np.all(wall.normal_vec == 0):
        raise ValueError("The Normalvector is the zero vector")
    return np.array(v - 2 * np.dot(v, n) * n)


def get_bbox_pts(wall: Wall) -> npdarr:
    """
    Return the LLF RBH Points of the Wall

    Parameter
        wall (Wall): The wall where the points are beeing calculated from

    Return:
        np.array([min_coords, max_coords]): returns a Vector of the Lowest Points and the highest Points of the Wall
    """
    points = np.array([wall.pos, wall.plains_vec, wall.plains_vec_2])
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    return np.array([min_coords, max_coords])


def plotWall(
    wall: Wall,
    ax: Axes,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
):
    """
    Plotts the given Wall on the given Axis

    Parameter:
        wall (Wall): The Wall that is beeing plotted
        ax (Axes): The Axes the Wall is beeing plotted on
        x_min (float): The minimum Value the x-Coordinates are plotted
        x_max (float): The maximum Value the x-Coordinates are plotted
        y_min (float): The minimum Value the y-Coordinates are plotted
        y_max (float): The maximum Value the y-Coordinates are plotted
        z_min (float): The minimum Value the z-Coordinates are plotted
        z_max (float): The maximum Value the z-Coordinates are plotted
    """
    # Sets the Limit to the given Parameter or the BBoxes (The one that Limits more)
    x_min = max(y_min, wall.min_cords[0])
    x_max = min(y_max, wall.max_cords[0])
    y_min = max(y_min, wall.min_cords[1])
    y_max = min(y_max, wall.max_cords[1])
    z_min = max(z_min, wall.min_cords[2])
    z_max = min(z_max, wall.max_cords[2])
    if wall.normal_vec[2] != 0:
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(x_min, x_max, 100)
        x, y = np.meshgrid(x, y)
        # Calculates z based on the multiple Values of x,y with the Formular for Plains transformed to:
        # z = (-d - a * x -b * y) / c
        z = (
            -wall.distance_origin - wall.normal_vec[0] * x - wall.normal_vec[1] * y
        ) / wall.normal_vec[2]
        # Saves all Values within the boundries set by the parameters and the BBox
        mask = (
            (z >= z_min)
            & (z <= z_max)
            & (z >= wall.min_cords[2])
            & (z <= wall.max_cords[2])
        )
        z = np.ma.masked_where(~mask, z)
        # if there aren't any valid values stop plotting
        if not np.any(mask):
            return
        # Plots the Wall on the Axis
        ax.plot_surface(x, y, z, alpha=wall.opacity, color=wall.color)
    elif wall.normal_vec[1] != 0:
        x = np.linspace(x_min, x_max, 100)
        z = np.linspace(z_min, z_max, 100)
        x, z = np.meshgrid(x, z)
        # If the normal vector z-Coordinate is zero it isn't possible to calculate z, because a division with 0 is caused
        # The plains formular is then transformed to:
        #  y = (-d - a * x) / b
        y = (-wall.normal_vec[0] * x - wall.distance_origin) / wall.normal_vec[1]
        # Saves all Values within the boundries set by the parameters and the BBox
        mask = (
            (y >= y_min)
            & (y <= y_max)
            & (y >= wall.min_cords[1])
            & (y <= wall.max_cords[1])
        )
        y = np.ma.masked_where(~mask, y)
        # if there aren't any valid values stop plotting
        if not np.any(mask):
            return
        # Plots the Wall on the Axis
        ax.plot_surface(x, y, z, alpha=wall.opacity, color=wall.color)
    elif wall.normal_vec[0] != 0:
        y = np.linspace(y_min, y_max, 100)
        z = np.linspace(z_min, z_max, 100)
        y, z = np.meshgrid(y, z)
        # If the normal vektor only has a value that is not 0 in the x-Coordinate
        # The formular for planes is then transformed to:
        # x = -d / a
        x = np.full_like(y, -wall.distance_origin / wall.normal_vec[0])
        mask = (
            (x >= x_min)
            & (x <= x_max)
            & (x >= wall.min_cords[0])
            & (x <= wall.max_cords[0])
        )
        x = np.ma.masked_where(~mask, x)
        # if there aren't any valid values stop plotting
        if not np.any(mask):
            return
        # Plots the Wall on the Axis
        ax.plot_surface(x, y, z, alpha=wall.opacity, color=wall.color)
    # If the normal vector is the zero vector sent an error 
    elif np.all(wall.normal_vec == 0):
        sys.exit("The Normalen Vector which was calculated is the Zero Vector")


if __name__ == "__main__":
    print("Testing the Wall class")
    wal = Wall(
        pos=np.array([1, 0, 0]),
        plains_vec=np.array([1, 1, 4]),
        plains_vec_2=np.array([0, 1, 3]),
    )
    print(wal)
    print("Bounding box points " + wal.bbox.__str__())
