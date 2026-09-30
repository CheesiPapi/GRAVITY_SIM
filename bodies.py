# we make the bodies in this file
import physics
import numpy as np

class Planet:
    def __init__(self, mass, radius, position, velocity):
        self.mass = mass
        self.radius = radius
        # We conver lists to NumPy arrays with float values for presision
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.bounce = False
        self.collision_response = None
        self.dynamic_collision = False
        self.force = np.array([0.0, 0.0, 0.0])
        self.bodies = []
        self.physics = physics.Physics()