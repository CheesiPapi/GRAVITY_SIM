import vtk


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

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.05, 0.05, 0.05)

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1000, 800)
        self.render_window.SetWindowName("Gravity Simulator")

        self.sync_actors()
        self.renderer.ResetCamera()

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleUser())  # mouse no longer moves the camera
        self.interactor.Initialize()
        self.interactor.AddObserver("TimerEvent", self.on_timer)
        self.interactor.AddObserver("KeyPressEvent", self.on_key)
        self.interactor.AddObserver("MouseWheelForwardEvent", lambda o, e: self.zoom(1.1))
        self.interactor.AddObserver("MouseWheelBackwardEvent", lambda o, e: self.zoom(1 / 1.1))
        self.interactor.CreateRepeatingTimer(16)

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

    def set_bodies(self, bodies):
        """Swap in a whole new set of bodies (used by the console)."""
        self.sim.bodies = bodies
        self.sim.events.clear()
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
        for hook in self.tick_hooks:
            hook()
        self.advance()
        self.sync_actors()
        self.update_actors()

        for ev in self.sim.pop_events():
            self.flashes.append(Flash(self.renderer, ev["point"], ev["size"], ev["shatter"]))
        self.flashes = [f for f in self.flashes if f.update()]

        self.render_window.Render()

    # ---------- camera controls ----------
    def on_key(self, obj, event):
        key = obj.GetKeySym()
        camera = self.renderer.GetActiveCamera()

        if key == "Left":
            camera.Azimuth(self.ROTATE_STEP)
        elif key == "Right":
            camera.Azimuth(-self.ROTATE_STEP)
        elif key == "Up":
            camera.Elevation(self.ROTATE_STEP)
            camera.OrthogonalizeViewUp()
        elif key == "Down":
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