from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray


from shapely.geometry import Point
from shapely.prepared import prep


if TYPE_CHECKING:
    from diatom.diatom import Diatom


class DiatomGrid():
    def __init__(self, diatom: "Diatom"):
        self.diatom = diatom
        self.points: NDArray[np.floating] = np.array([]) # shape: (numPoints**2, 2)
        self.feasible_points: NDArray[np.bool] = np.array([])
        self.grid_dimensions: NDArray[np.floating] = np.array([0,0])
        self.points_per_axis: tuple[int, int] = (0,0)


    def sample_polytope(self, delta: float = 0.1) -> None:
        poly = self.diatom.analyze.polytope
        prepared_poly = prep(poly)

        minx, miny, maxx, maxy = poly.bounds
        interval_x = maxx - minx
        interval_y = maxy - miny
        self.grid_dimensions = np.array([interval_x, interval_y])

        dx, dy = delta * interval_x, delta * interval_y
        xs = np.arange(minx, maxx + dx, dx)
        ys = np.arange(miny, maxy + dy, dy)
        self.points_per_axis = (len(xs), len(ys))

        grid_x, grid_y = np.meshgrid(xs, ys)
        points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        self.points = points 

        covers_xy = getattr(prepared_poly, "covers_xy", None)

        if covers_xy is not None:
            feasible = np.fromiter(
                (covers_xy(x, y) for x, y in points),
                dtype=bool,
                count=len(points)
            )
        else:
            feasible = np.fromiter(
                (prepared_poly.covers(Point(x, y)) for x, y in points),
                dtype=bool,
                count=len(points)
                )

        self.feasible_points = feasible



    