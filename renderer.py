import numpy as np
import vtk

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
            self.renderer.AddActor(actor)
            self.actors.append(actor)
            self.actors[0].SetPosition(body.position[0], body.position[1], body.position[2])
            
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