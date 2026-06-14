import vtk
from physics import update_physics

class SimulationRenderer:
    def __init__(self, bodies, dt):
        self.bodies = bodies
        self.dt = dt
        
        # 1. Setup the basic VTK Window
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.05, 0.05, 0.05) # Deep space gray
        
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1000, 800)
        self.render_window.SetWindowName("Gravity Simulator")
        
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        
        # 2. Build the VTK Actors (The Visuals)
        self.actors = [] # We keep a list of the 3D shapes so we can move them later
        
        for body in self.bodies:
            # Create the 3D sphere based on the planet's actual radius
            source = vtk.vtkSphereSource()
            source.SetRadius(body.radius)
            source.SetPhiResolution(20)
            source.SetThetaResolution(20)
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(source.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            # Place it at the planet's starting XYZ coordinates
            actor.SetPosition(body.position[0], body.position[1], body.position[2])
            
            # Make the heavy sun Yellow, and the smaller planets Blue
            if body.mass > 1000:
                actor.GetProperty().SetColor(1.0, 0.9, 0.1) # Sun
            else:
                actor.GetProperty().SetColor(0.2, 0.5, 1.0) # Planet
            
            self.renderer.AddActor(actor)
            self.actors.append(actor)
            
        # 3. Create the Simulation Loop (The Heartbeat)
        self.interactor.AddObserver('TimerEvent', self.update_frame)
      
        
    def update_frame(self, obj, event):
        # Step 1: Run the math to find new positions
        update_physics(self.bodies, self.dt, G=1000) # We use a stronger G for more dramatic motion in our small universe
        
        # Step 2: Move the VTK actors to match the new math positions
        for i, body in enumerate(self.bodies):
            self.actors[i].SetPosition(body.position[0], body.position[1], body.position[2])
            
        # Step 3: Tell the window to draw the new frame
        self.render_window.Render()
        
    def start(self):
        # 1. Turn the engine on FIRST
        self.interactor.Initialize() 
        
        # 2. NOW we can safely start the heartbeat clock
        self.interactor.CreateRepeatingTimer(16) 
        
        # 3. Open the window and begin!
        self.render_window.Render()
        self.interactor.Start()