import time

import numpy as np
import vtk

from hud import Overlay
from vectors import VectorField

VECTOR_MODES = ("off", "vel", "acc", "both")


class Flash:
    """A short-lived expanding, fading sphere drawn where a collision happened."""
    FRAMES = 25

    def __init__(self, renderer, point, size, shatter):
        self.renderer = renderer
        self.age = 0
        self.max_radius = size * (3.0 if shatter else 1.5)

        source = vtk.vtkSphereSource()
        source.SetRadius(1.0)
        source.SetThetaResolution(24)
        source.SetPhiResolution(24)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())

        self.actor = vtk.vtkActor()
        self.actor.SetMapper(mapper)
        prop = self.actor.GetProperty()
        if shatter:
            prop.SetColor(1.0, 0.6, 0.2)
        else:
            prop.SetColor(1.0, 1.0, 0.85)
        prop.LightingOff()                      # glow, ignore scene lighting
        prop.SetOpacity(0.9)
        self.actor.SetPosition(*point)
        self.actor.SetScale(0.01, 0.01, 0.01)
        renderer.AddActor(self.actor)

    def update(self):
        """Advance one frame. Returns False once the flash is finished."""
        self.age += 1
        t = self.age / self.FRAMES
        if t >= 1.0:
            self.renderer.RemoveActor(self.actor)
            return False
        radius = self.max_radius * (1 - (1 - t) ** 2)      # fast start, slow end
        self.actor.SetScale(radius, radius, radius)
        self.actor.GetProperty().SetOpacity(0.9 * (1 - t))
        return True


class SimulationRenderer:
    ROTATE_STEP = 3.0   # degrees per arrow-key press (holding a key repeats)

    def __init__(self, sim, dt):
        self.sim = sim
        self.dt = dt
        self.speed = 1.0          # >1: extra physics steps per tick, <1: smaller dt
        self.paused = False
        self.tick_hooks = []      # callables run at the start of every tick
        self.actors = {}          # Body -> vtkActor
        self.flashes = []
        self.console = None

        self.vector_mode = "vel"  # off | vel | acc | both
        self.vector_scale = 1.0   # multiplier on the default arrow lengths

        self.fps = 60.0           # smoothed, shown in the HUD
        self.physics_ms = 0.0
        self._last_tick = None

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.05, 0.05, 0.05)

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1000, 800)
        self.render_window.SetWindowName("Gravity Simulator")

        # arrow lengths = magnitude * scale (tuned for G ~ 200; adjust with 'vectors scale')
        self.vel_arrows = VectorField(self.renderer, color=(0.2, 0.9, 1.0), scale=0.2)
        self.acc_arrows = VectorField(self.renderer, color=(1.0, 0.35, 0.3), scale=0.05)
        self.overlay = Overlay(self)

        self.sync_actors()
        self.renderer.ResetCamera()

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        style = vtk.vtkInteractorStyleUser()      # mouse no longer moves the camera
        # Swallow VTK's built-in letter shortcuts (q/e = exit, w = wireframe,
        # r = reset camera, ...) so typing commands can't trigger them.
        style.AddObserver("CharEvent", lambda o, e: None)
        self.interactor.SetInteractorStyle(style)
        self.interactor.Initialize()
        self.interactor.AddObserver("TimerEvent", self.on_timer)
        self.interactor.AddObserver("KeyPressEvent", self.on_key)
        self.interactor.AddObserver("MouseWheelForwardEvent", lambda o, e: self.zoom(1.1))
        self.interactor.AddObserver("MouseWheelBackwardEvent", lambda o, e: self.zoom(1 / 1.1))
        self.interactor.CreateRepeatingTimer(16)

    def attach_console(self, console):
        self.console = console
        self.overlay.console = console
        self.tick_hooks.append(console.tick)

    # ---------- actors ----------
    def _make_actor(self, body):
        source = vtk.vtkSphereSource()
        source.SetRadius(body.radius)
        source.SetThetaResolution(32)
        source.SetPhiResolution(32)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*body.color)
        actor.SetPosition(*body.position)
        return actor

    def sync_actors(self):
        """Make the set of actors match sim.bodies (bodies come and go on merges)."""
        live = set(self.sim.bodies)
        for body in [b for b in self.actors if b not in live]:
            self.renderer.RemoveActor(self.actors.pop(body))
        for body in self.sim.bodies:
            if body not in self.actors:
                actor = self._make_actor(body)
                self.renderer.AddActor(actor)
                self.actors[body] = actor

    def update_actors(self):
        for body, actor in self.actors.items():
            actor.SetPosition(*body.position)

    def update_vectors(self):
        show_vel = self.vector_mode in ("vel", "both")
        show_acc = self.vector_mode in ("acc", "both")
        bodies = self.sim.bodies
        self.vel_arrows.set_visible(show_vel and bodies)
        self.acc_arrows.set_visible(show_acc and bodies)
        if not bodies or not (show_vel or show_acc):
            return

        pos = np.array([b.position for b in bodies])
        rad = np.array([b.radius for b in bodies])
        if show_vel:
            vel = np.array([b.velocity for b in bodies])
            self.vel_arrows.update(pos, vel, rad, self.vector_scale)
        if show_acc:
            acc = np.array([b.force / b.mass for b in bodies])
            self.acc_arrows.update(pos, acc, rad, self.vector_scale)

    def set_bodies(self, bodies):
        """Swap in a whole new set of bodies (used by the console)."""
        self.sim.bodies = bodies
        self.sim.events.clear()
        self.sim.time = 0.0
        self.sim.step_count = 0
        self.sync_actors()
        self.renderer.ResetCamera()
        self.render_window.Render()

    # ---------- simulation tick ----------
    def advance(self):
        if self.paused:
            return
        if self.speed >= 1.0:
            for _ in range(int(round(self.speed))):
                self.sim.step(self.dt)
        else:
            self.sim.step(self.dt * self.speed)

    def on_timer(self, obj, event):
        now = time.perf_counter()
        if self._last_tick is not None:
            self.fps = 0.9 * self.fps + 0.1 / max(now - self._last_tick, 1e-6)
        self._last_tick = now

        for hook in self.tick_hooks:
            hook()

        start = time.perf_counter()
        self.advance()
        self.physics_ms = 0.9 * self.physics_ms + 0.1 * (time.perf_counter() - start) * 1000.0

        self.sync_actors()
        self.update_actors()
        self.update_vectors()

        for ev in self.sim.pop_events():
            self.flashes.append(Flash(self.renderer, ev["point"], ev["size"], ev["shatter"]))
        self.flashes = [f for f in self.flashes if f.update()]

        self.overlay.update()
        self.render_window.Render()

    # ---------- keyboard / camera ----------
    def on_key(self, obj, event):
        keysym = obj.GetKeySym()
        if self.console is not None and self.console.handle_key(keysym, obj.GetKeyCode()):
            return                      # typing a command, or a console hotkey

        camera = self.renderer.GetActiveCamera()
        if keysym == "Left":
            camera.Azimuth(self.ROTATE_STEP)
        elif keysym == "Right":
            camera.Azimuth(-self.ROTATE_STEP)
        elif keysym == "Up":
            camera.Elevation(self.ROTATE_STEP)
            camera.OrthogonalizeViewUp()
        elif keysym == "Down":
            camera.Elevation(-self.ROTATE_STEP)
            camera.OrthogonalizeViewUp()
        else:
            return

        self.renderer.ResetCameraClippingRange()
        self.render_window.Render()

    def zoom(self, factor):
        self.renderer.GetActiveCamera().Dolly(factor)
        self.renderer.ResetCameraClippingRange()

    def start(self):
        self.render_window.Render()
        self.interactor.Start()