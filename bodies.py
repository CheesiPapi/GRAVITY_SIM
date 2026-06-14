# we make the bodies in this file

import numpy as np

class Planet:
    def __init__(self, mass, radius, position, velocity):
        self.mass = mass
        self.radius = radius
        # We conver lists to NumPy arrays with float values for presision
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)