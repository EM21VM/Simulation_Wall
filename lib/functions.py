import numpy as np
import math as math
from lib.constants import npdarr


def normalize(v: npdarr) -> npdarr:
    if not np.any(v):
        raise ValueError("Can't normalize zero vector")
    return v / np.linalg.norm(v)


def dist_sqr(vec1: npdarr, vec2: npdarr) -> np.float64:
    return np.dot(vec1 - vec2, vec1 - vec2)


def distance(vec1: npdarr, vec2: npdarr) -> np.float64:
    return np.linalg.norm(vec1 - vec2)


def value_vec(vec1: npdarr) -> float:
    return abs(
        math.sqrt(math.pow(vec1[0], 2) + math.pow(vec1[1], 2) + math.pow(vec1[2], 2))
    )


def calc_distance(point: npdarr, normal_vec: npdarr, distance_origin: float) -> float:
    """
    Return the distance between a point and a plane

    Parameter:
        point (npdarr): The current position of the point
        normal_vec (npdarr): The normal vector of the plane
        distance_origin (npdarr) : The distance from the origin 
        
    Returns: 
        distance (float): The distance between a point and a plane
    """
    normal_vec_norm: float = np.linalg.norm(normal_vec)
    if not np.any(normal_vec):
        print("Can't calculate distance with the zero vector as the normal vector")
        # The Distance formular for a plane in  Coordinate shape is:
        # d = |n1 * p1 + n2 * p2 + n3 * p3 + k|/sqrt(n1^2 + n2^2 + n3^2)
        # p being the position of the point
        # n being the normal vector of the plane
        # k the distance to origin
    return np.abs(np.dot(normal_vec, point) + distance_origin) / normal_vec_norm


if __name__ == "__main__":
    v: npdarr = np.array([3, -4])
    print(f"{v} -> {normalize(v)}")
    a = np.array([2, 2, 2])
    erg = value_vec(a)
    print(erg)
