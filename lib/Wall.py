import numpy as np
import sys
import os
import time as time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib.Particle import Particle
from lib.constants import ZERO_VEC, npdarr, Axes
from lib.Object import Object
from lib.functions import cross_vec


class Wall(Object):
    "A Wall in a 3-Dimensinal Space"

    def __init__(
        # Formula for plains
        # a*x + b*y + c*z + d = 0
        # normal_vec[0]*x + normal_vec[1]*y + normal_vec[2]*z + distance_origin = 0
        self,
        pos: npdarr = np.copy(ZERO_VEC),
        plains_vec: npdarr = np.copy(ZERO_VEC),
        plains_vec_2: npdarr = np.copy(ZERO_VEC),
        norm_vec: npdarr = np.array([ZERO_VEC]),
        dis_origin: float = 0,
        offset: float = 0,
    ) -> None:
        print(f"start_pos: {pos}")
        self.pos = pos
        self.plains_vec = plains_vec
        self.plains_vec_2 = plains_vec_2
        print("P1: " + str(plains_vec) + " P2: " + str(plains_vec_2))
        print("AB: " + str(pos - plains_vec) + "AC: " + str(pos - plains_vec_2))
        self.normal_vec = cross_vec(plains_vec - pos, plains_vec_2 - pos)
        self.normal_vec = self.normal_vec / np.linalg.norm(self.normal_vec)
        self.distance_origin = -pos.dot(self.normal_vec)
        if not np.all(norm_vec == 0) and dis_origin != 0:
            self.normal_vec = norm_vec
            self.distance_origin = dis_origin
        self.offset = offset
        print("Normalvector: " + str(self.normal_vec))
        print("Distance to origin: " + str(self.distance_origin))
        bbox_pts = get_bbox_pts(self)
        print("Bounding box points " + str(bbox_pts))
        super().__init__(pos, bbox_pts=bbox_pts)


def wall_collision(particle: Particle, wall: Wall) -> npdarr:
    v = particle.vel
    n = wall.normal_vec
    if np.all(wall.normal_vec == 0):
        print("The velocity is the zero vector")
    # print(
    #     "Vel: "
    #     + str(v)
    #     + "Normalvector: "
    #     + str(n)
    #     + "NEW Vel: "
    #     + str(v - 2 * np.dot(v, n) * n)
    # )
    return np.array(v - 2 * np.dot(v, n) * n)


def get_bbox_pts(wall: Wall) -> npdarr:
    offset = wall.offset
    points = np.array([wall.pos, wall.plains_vec, wall.plains_vec_2])
    min_coords = points.min(axis=0) - offset
    max_coords = points.max(axis=0)

    return np.array([min_coords, max_coords])


def get_bbox_pts_n(wall: Wall) -> npdarr:
    offset = wall.offset
    normal_vec = wall.normal_vec
    if np.isclose(normal_vec[0], 0) and np.isclose(normal_vec[1], 0):
        perp1 = np.array([1, 0, 0])
    else:
        perp1 = np.cross(normal_vec, np.array([0, 0, 1]))
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(normal_vec, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)

    perp1 *= offset
    perp2 *= offset

    corners = [
        wall.pos + perp1 + perp2,
        wall.pos + perp1 - perp2,
        wall.pos - perp1 + perp2,
        wall.pos - perp1 - perp2,
    ]
    corners = np.array(corners)
    min_coords = corners.min(axis=0)
    max_coords = corners.max(axis=0)
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
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(x_min, x_max, 100)
    x, y = np.meshgrid(x, y)
    if wall.normal_vec[2] != 0:
        z = (
            -wall.distance_origin - wall.normal_vec[0] * x - wall.normal_vec[1] * y
        ) / wall.normal_vec[2]
        # z = np.clip(z, z_min, z_max)
        mask = (z >= z_min) & (z <= z_max)
        z = np.ma.masked_where(~mask, z)
        if not np.any(mask):
            return
        ax.plot_surface(x, y, z, alpha=0.7)
    elif wall.normal_vec[1] != 0:
        x = np.linspace(x_min, x_max, 100)
        z = np.linspace(z_min, z_max, 100)
        X, Z = np.meshgrid(x, z)
        Y = (-wall.normal_vec[0] * x - wall.distance_origin) / wall.normal_vec[1]
        # Y = np.clip(Y, y_min, y_max)
        mask = (Y >= y_min) & (Y <= y_max)
        Y = np.ma.masked_where(~mask, Y)
        if not np.any(mask):
            return
        ax.plot_surface(X, Y, Z, alpha=0.7)
    elif wall.normal_vec[0] != 0:
        y = np.linspace(y_min, y_max, 100)
        z = np.linspace(z_min, z_max, 100)
        y, z = np.meshgrid(y, z)
        x = np.full_like(x, -wall.distance_origin / wall.normal_vec[0])
        # x = np.clip(x, x_min, x_max)
        mask = (x >= x_min) & (x <= x_max)
        x = np.ma.masked_where(~mask, x)
        if not np.any(mask):
            return
        ax.plot_surface(x, y, z, alpha=0.7)
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
