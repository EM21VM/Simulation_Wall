import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib.constants import ZERO_VEC, npdarr
from lib.Object import Object


class Wall(Object):
    "A Wall in a 3-Dimensinal Space"

    def __init__(
        self,
        pos: npdarr = np.copy(ZERO_VEC),
        end_pos: npdarr = np.copy(ZERO_VEC),
        color: str = "#FF0000",
        opacity: float = 1.0,
    ) -> None:
        print(f"start_pos: {pos}")
        self.width: float = abs(pos[0] - end_pos[0])
        self.height: float = abs(pos[1] - end_pos[1])
        first_bbox_pts = np.array(pos)
        last_bbox_pts = np.array([pos[0] + self.width, pos[1] + self.height, pos[2]])
        bbox_pts: npdarr = np.array([first_bbox_pts, last_bbox_pts])
        super().__init__(pos, bbox_pts=bbox_pts, color=color, opacity=opacity)


if __name__ == "__main__":
    print("Testing the Wall class")
    wal = Wall(
        pos=np.array([1, 1, 0]),
        end_pos=np.array([2, 2, 0]),
        color="#FF0000",
        opacity=1,
    )
    print(wal)
    print("Bounding box points " + wal.bbox.__str__())
