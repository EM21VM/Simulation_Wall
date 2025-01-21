import numpy as np
import sys
import os
import time as time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib.Particle import Particle
from lib.constants import ZERO_VEC, npdarr, Axes
from lib.Object import Object
from lib.functions import cross_vec, skalarproduct


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
        offset: float = 0,
    ) -> None:
        print(f"start_pos: {pos}")
        self.pos = pos
        self.plains_vec = plains_vec
        self.plains_vec_2 = plains_vec_2
        print("P1: " + str(plains_vec) + " P2: " + str(plains_vec_2))
        print("AB: " + str(pos - plains_vec) + "AC: " + str(pos - plains_vec_2))
        self.normal_vec = cross_vec(pos - plains_vec, pos - plains_vec_2)
        self.distance_origin = -pos.dot(self.normal_vec)
        self.offset = offset
        print("Normalvector: " + str(self.normal_vec))
        print("Distance to origin: " + str(self.distance_origin))
        bbox_pts = get_bbox_pts(self)
        print("Bounding box points " + str(bbox_pts))
        super().__init__(pos, bbox_pts=bbox_pts)


def wall_collision(p1: Particle, w1: Wall) -> npdarr:
    v = p1.vel
    n = w1.normal_vec
    # print("Vel: " + str(v) + "Normalvector: " + str(n) + "NEW Vel: " + str(v - skalarproduct(v,n) * n))
    return np.array(v - 2 * v.dot(n) * n)


def get_bbox_pts(wall: Wall):
    offset = wall.offset
    if wall.pos[0] < wall.plains_vec[0] and wall.pos[0] < wall.plains_vec_2[0]:
        first_bbox_x_pt = wall.pos[0]
    elif wall.plains_vec[0] < wall.pos[0] and wall.plains_vec[0] < wall.plains_vec_2[0]:
        first_bbox_x_pt = wall.plains_vec[0]
    else:
        first_bbox_x_pt = wall.plains_vec_2[0]

    if wall.pos[1] < wall.plains_vec[1] and wall.pos[1] < wall.plains_vec_2[1]:
        first_bbox_y_pt = wall.pos[1]
    elif wall.plains_vec[1] < wall.pos[1] and wall.plains_vec[1] < wall.plains_vec_2[1]:
        first_bbox_y_pt = wall.plains_vec[1]
    else:
        first_bbox_y_pt = wall.plains_vec_2[1]

    if wall.pos[2] < wall.plains_vec[2] and wall.pos[2] < wall.plains_vec_2[2]:
        first_bbox_z_pt = wall.pos[2]
    elif wall.plains_vec[2] < wall.pos[2] and wall.plains_vec[2] < wall.plains_vec_2[2]:
        first_bbox_z_pt = wall.plains_vec[2]
    else:
        first_bbox_z_pt = wall.plains_vec_2[2]
    first_bbox_pts = [
        first_bbox_x_pt - offset,
        first_bbox_y_pt - offset,
        first_bbox_z_pt - offset,
    ]

    if wall.pos[0] > wall.plains_vec[0] and wall.pos[0] > wall.plains_vec_2[0]:
        last_bbox_x_pt = wall.pos[0]
    elif wall.plains_vec[0] > wall.pos[0] and wall.plains_vec[0] > wall.plains_vec_2[0]:
        last_bbox_x_pt = wall.plains_vec[0]
    else:
        last_bbox_x_pt = wall.plains_vec_2[0]

    if wall.pos[1] > wall.plains_vec[1] and wall.pos[1] > wall.plains_vec_2[1]:
        last_bbox_y_pt = wall.pos[1]
    elif wall.plains_vec[1] > wall.pos[1] and wall.plains_vec[1] > wall.plains_vec_2[1]:
        last_bbox_y_pt = wall.plains_vec[1]
    else:
        last_bbox_y_pt = wall.plains_vec_2[1]

    if wall.pos[2] > wall.plains_vec[2] and wall.pos[2] > wall.plains_vec_2[2]:
        last_bbox_z_pt = wall.pos[2]
    elif wall.plains_vec[2] > wall.pos[2] and wall.plains_vec[2] > wall.plains_vec_2[2]:
        last_bbox_z_pt = wall.plains_vec[2]
    else:
        last_bbox_z_pt = wall.plains_vec_2[2]
    last_bbox_pts = [
        last_bbox_x_pt + offset,
        last_bbox_y_pt + offset,
        last_bbox_z_pt + offset,
    ]
    return np.array([first_bbox_pts, last_bbox_pts])


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
        print("1")
        z = (
            -wall.distance_origin - wall.normal_vec[0] * x - wall.normal_vec[1] * y
        ) / wall.normal_vec[2]
        z = np.clip(z, z_min, z_max)
        ax.plot_surface(x, y, z, alpha=0.7)
        ax.scatter(
            x=[wall.bbox.pts[0][0], wall.bbox.pts[1][0]],
            y=[wall.bbox.pts[0][1], wall.bbox.pts[1][1]],
            z=[wall.bbox.pts[0][2], wall.bbox.pts[1][2]],
        )
    elif wall.normal_vec[1] != 0:
        print("2")
        x = np.linspace(x_min, x_max, 100)
        z = np.linspace(z_min, z_max, 100)
        X, Z = np.meshgrid(x, z)
        Y = (-wall.normal_vec[0] * x - wall.distance_origin) / wall.normal_vec[1]
        Y = np.clip(Y, y_min, y_max)
        ax.plot_surface(X, Y, Z, alpha=0.7)
    elif wall.normal_vec[0] != 0:
        print("3")
        y = np.linspace(y_min, y_max, 100)
        z = np.linspace(z_min, z_max, 100)
        y, z = np.meshgrid(y, z)
        x = np.full_like(x, -wall.distance_origin / wall.normal_vec[0])
        x = np.clip(x, x_min, x_max)
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