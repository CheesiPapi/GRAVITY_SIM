import vtk

class SimulationRenderer:
    def __init__(self, sim, dt):
        self.sim = sim
        self.dt = dt

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.05, 0.05, 0.05)

        # build one actor per body, in the same order
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

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1000, 800)
        self.render_window.SetWindowName("Gravity Simulator")

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.Initialize()
        self.interactor.AddObserver("TimerEvent", self.on_timer)
        self.interactor.CreateRepeatingTimer(16)

    def on_timer(self, obj, event):
        self.sim.step(self.dt)
        self.update_actors()
        self.render_window.Render()

    def update_actors(self):
        for actor, body in zip(self.actors, self.sim.bodies):
            actor.SetPosition(*body.position)

    def start(self):
        self.render_window.Render()
        self.interactor.Start()