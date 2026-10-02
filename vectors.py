"""Arrows drawn from each body (velocity, acceleration, ...).

All arrows for one vector type live in a single vtkPolyData / actor, so adding
more bodies doesn't add more actors. Each arrow is 6 points and 5 line
segments: a shaft (tail -> tip) and four barbs forming the arrowhead.
"""
import numpy as np
import vtk
from vtk.util import numpy_support


class VectorField:
    def __init__(self, renderer, color, scale):
        self.scale = scale            # world units of arrow length per unit of vector
        self._count = -1              # number of arrows the line cells were built for

        self.points = vtk.vtkPoints()
        self.poly = vtk.vtkPolyData()
        self.poly.SetPoints(self.points)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.poly)
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        prop = self.actor.GetProperty()
        prop.SetColor(*color)
        prop.SetLineWidth(2.0)
        prop.LightingOff()
        renderer.AddActor(self.actor)

    def set_visible(self, visible):
        self.actor.SetVisibility(bool(visible))

    def _rebuild_cells(self, n):
        lines = vtk.vtkCellArray()
        for i in range(n):
            base = 6 * i
            for end in (1, 2, 3, 4, 5):           # shaft, then 4 barbs
                a, b = (base, base + 1) if end == 1 else (base + 1, base + end)
                lines.InsertNextCell(2)
                lines.InsertCellPoint(a)
                lines.InsertCellPoint(b)
        self.poly.SetLines(lines)
        self._count = n

    def update(self, positions, vectors, radii, user_scale=1.0):
        n = len(positions)
        if n == 0:
            return
        if n != self._count:
            self._rebuild_cells(n)

        length = np.linalg.norm(vectors, axis=1)
        nonzero = (length > 1e-12)[:, None]
        d = np.divide(vectors, length[:, None], out=np.zeros_like(vectors), where=nonzero)

        arrow_len = length * self.scale * user_scale
        tail = positions + d * radii[:, None]            # start at the sphere's surface
        tip = tail + d * arrow_len[:, None]

        # two unit vectors perpendicular to the arrow, for the arrowhead barbs
        ref = np.where(np.abs(d[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0]])
        u = np.cross(d, ref)
        un = np.linalg.norm(u, axis=1)[:, None]
        u = np.divide(u, un, out=np.zeros_like(u), where=un > 1e-12)
        w = np.cross(d, u)

        head = (0.25 * arrow_len)[:, None]
        back = tip - d * head
        side = 0.4 * head

        pts = np.empty((n, 6, 3))
        pts[:, 0] = tail
        pts[:, 1] = tip
        pts[:, 2] = back + u * side
        pts[:, 3] = back - u * side
        pts[:, 4] = back + w * side
        pts[:, 5] = back - w * side

        self.points.SetData(numpy_support.numpy_to_vtk(pts.reshape(-1, 3), deep=True))
        self.points.Modified()
        self.poly.Modified()