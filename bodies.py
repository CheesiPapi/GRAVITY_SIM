import numpy as np

class Body:
    def __init__(self, position, velocity=(0, 0, 0), mass=1.0, radius=1.0,
                 color=(0.7, 0.7, 0.7)):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = float(mass)
        self.radius = float(radius)
        self.color = color
        self.force = np.zeros(3)