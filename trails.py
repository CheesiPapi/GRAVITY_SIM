"""Fading orbit trails, drawn as one polyline per body in a single actor.

History is kept in a (bodies, length, 3) array that is shifted each sample.
When bodies merge, shatter, get culled or a new scenario starts, the rows are
re-mapped: survivors keep their history, new bodies start with an empty trail.
"""
import numpy as np
import vtk
from vtk.util import numpy_support


class Trails:
    def __init__(self, renderer, length=120):
        self.length = length
        self.enabled = True
        self._bodies = []
        self._buf = np.zeros((0, length, 3))
        self._count = np.zeros(0, dtype=int)

        self.points = vtk.vtkPoints()
        self.poly = vtk.vtkPolyData()
        self.poly.SetPoints(self.points)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.poly)
        mapper.SetScalarModeToUsePointData()
        mapper.SetColorModeToDirectScalars()
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        prop = self.actor.GetProperty()
        prop.SetLineWidth(1.5)
        prop.LightingOff()
        renderer.AddActor(self.actor)

    # ---------- control ----------
    def set_enabled(self, on):
        self.enabled = bool(on)
        self.actor.SetVisibility(self.enabled)
        if not self.enabled:
            self.clear()

    def set_length(self, length):
        length = int(max(2, min(length, 2000)))
        keep = min(length, self.length)
        buf = np.zeros((len(self._bodies), length, 3))
        buf[:, length - keep:] = self._buf[:, self.length - keep:]
        self._buf = buf
        self._count = np.minimum(self._count, keep)
        self.length = length

    def clear(self):
        self._bodies = []
        self._buf = np.zeros((0, self.length, 3))
        self._count = np.zeros(0, dtype=int)
        self._show(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8), np.array([0]))

    # ---------- per frame ----------
    def _same_bodies(self, bodies):
        return (len(bodies) == len(self._bodies)
                and all(a is b for a, b in zip(self._bodies, bodies)))

    def _remap(self, bodies):
        row = {id(b): i for i, b in enumerate(self._bodies)}
        buf = np.zeros((len(bodies), self.length, 3))
        count = np.zeros(len(bodies), dtype=int)
        for i, b in enumerate(bodies):
            old = row.get(id(b))
            if old is not None:
                buf[i] = self._buf[old]
                count[i] = self._count[old]
        self._bodies = list(bodies)
        self._buf, self._count = buf, count

    def update(self, bodies, record=True):
        if not self.enabled:
            return
        if not self._same_bodies(bodies):
            self._remap(bodies)
        if not len(bodies):
            self._show(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8), np.array([0]))
            return

        if record:
            self._buf[:, :-1] = self._buf[:, 1:]
            self._buf[:, -1] = [b.position for b in bodies]
            self._count = np.minimum(self._count + 1, self.length)

        length = self.length
        count = np.where(self._count >= 2, self._count, 0)         # skip 0/1-point trails
        k = np.arange(length)[None, :] - (length - count[:, None])  # 0 = oldest valid point
        valid = k >= 0
        pts = self._buf[valid]

        fade = k[valid] / np.maximum(np.repeat(count - 1, count), 1)
        brightness = (0.04 + 0.96 * fade ** 1.5)[:, None]
        base = np.array([b.color for b in bodies], dtype=float)
        base = 0.4 + 0.6 * base                                    # keep dim bodies visible
        colors = (np.repeat(base, count, axis=0) * brightness * 255).astype(np.uint8)

        offsets = np.concatenate([[0], np.cumsum(count[count > 0])])
        self._show(pts, colors, offsets)

    def _show(self, pts, colors, offsets):
        npts = len(pts)
        self.points.SetData(numpy_support.numpy_to_vtk(
            np.ascontiguousarray(pts, dtype=float).reshape(-1, 3), deep=True))
        cells = vtk.vtkCellArray()
        if npts:
            try:
                cells.SetData(
                    numpy_support.numpy_to_vtkIdTypeArray(offsets.astype(np.int64), deep=True),
                    numpy_support.numpy_to_vtkIdTypeArray(np.arange(npts, dtype=np.int64), deep=True))
            except (TypeError, AttributeError):                    # older VTK: one cell per line
                for a, b in zip(offsets[:-1], offsets[1:]):
                    cells.InsertNextCell(int(b - a))
                    for p in range(a, b):
                        cells.InsertCellPoint(int(p))
        self.poly.SetLines(cells)
        scalars = numpy_support.numpy_to_vtk(
            np.ascontiguousarray(colors, dtype=np.uint8), deep=True,
            array_type=vtk.VTK_UNSIGNED_CHAR)
        scalars.SetName("Colors")
        self.poly.GetPointData().SetScalars(scalars)
        self.points.Modified()
        self.poly.Modified()