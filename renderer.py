import vtk


class SimulationRenderer:
    def __init__(self, sim, dt):
        self.sim = sim
        self.dt = dt
        self.speed = 1.0          # >1: extra physics steps per tick, <1: smaller dt
        self.paused = False
        self.tick_hooks = []      # callables run at the start of every tick
        self.actors = []

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.05, 0.05, 0.05)

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1000, 800)
        self.render_window.SetWindowName("Gravity Simulator")

        self.rebuild_actors()

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.Initialize()
        self.interactor.AddObserver("TimerEvent", self.on_timer)
        self.interactor.AddObserver("KeyPressEvent", self.on_key)
        self.interactor.AddObserver("MouseWheelForwardEvent", lambda o, e: self.zoom(1.1))
        self.interactor.AddObserver("MouseWheelBackwardEvent", lambda o, e: self.zoom(1/1.1))
        self.interactor.CreateRepeatingTimer(16)

    def rebuild_actors(self):
        """Throw away old actors and build one per body, in the same order."""
        for actor in self.actors:
            self.renderer.RemoveActor(actor)
        self.actors = []

        for body in self.sim.bodies:
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

            self.renderer.AddActor(actor)
            self.actors.append(actor)

        self.renderer.ResetCamera()

    def set_bodies(self, bodies):
        self.sim.bodies = bodies
        self.rebuild_actors()
        self.render_window.Render()

    ROTATE_STEP = 3.0   # degrees per key press (holding a key repeats)

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
        self.update_actors()
        self.render_window.Render()

    def update_actors(self):
        for actor, body in zip(self.actors, self.sim.bodies):
            actor.SetPosition(*body.position)

    def start(self):
        self.render_window.Render()
        self.interactor.Start()
