# physics file
import numpy as np

class Physics:
    def __init__(self, G=6.67430e-11):
        self.G = G
        self.force = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.position = np.array([0.0, 0.0, 0.0])
        self.mass = 1.0
        self.radius = 1.0
        self.bounce = False
        self.collision_response = None
        self.dynamic_collision = False
        self.bodies = []

    def dist_diff(self, other):
        diff = other.position - self.position
        return np.sqrt(np.sum(diff**2))

    def bounce(self):
        self.bounce = True
        self.collision_response = True
        self.dynamic_collision = True
        self.force = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.position = np.array([0.0, 0.0, 0.0])

    def handle_collision(self, other):
        if self.dist_diff(other) < self.radius + other.radius:
            self.dynamic_collision(self, other)
            self.bounce = True
            self.collision_response = True
            other.bounce = True
            other.collision_response = True
            return True
        return False
    
    def gravity_force(self, other):
        diff = other.position - self.position
        dist = np.sqrt(np.sum(diff**2))
        if dist == 0:
            return np.array([0.0, 0.0, 0.0])
        F = self.G * ((self.mass * other.mass) / dist**2)
        unit_direction = diff / dist
        force_vector = F * unit_direction
        return force_vector
    
    def update_physics(self, dt, G=6.67430e-11):
        total_force = np.array([0.0, 0.0, 0.0])
        for other in self.bodies:
            if self is other:
                continue
            force_vector = self.gravity_force(other)
            total_force += force_vector
        acceleration = total_force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

