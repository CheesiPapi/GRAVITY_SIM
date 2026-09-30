import numpy as np

class Simulation:
    def __init__(self, bodies, G=200.0, softening=0.1, restitution=1.0):
        self.bodies = bodies
        self.G = G
        self.softening = softening
        self.restitution = restitution

    def compute_forces(self):
        for b in self.bodies:
            b.force[:] = 0.0
        n = len(self.bodies)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.bodies[i], self.bodies[j]
                diff = b.position - a.position
                dist2 = diff @ diff + self.softening**2
                dist = np.sqrt(dist2)
                f = self.G * a.mass * b.mass / dist2
                force = f * diff / dist
                a.force += force      # equal and opposite
                b.force -= force

    def handle_collisions(self):
        n = len(self.bodies)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.bodies[i], self.bodies[j]
                diff = b.position - a.position
                dist = np.linalg.norm(diff)
                min_dist = a.radius + b.radius
                if dist >= min_dist or dist == 0.0:
                    continue

                normal = diff / dist
                inv_a, inv_b = 1.0 / a.mass, 1.0 / b.mass
                inv_sum = inv_a + inv_b

                # 1. separate the overlap (lighter body moves more)
                overlap = min_dist - dist
                a.position -= normal * overlap * inv_a / inv_sum
                b.position += normal * overlap * inv_b / inv_sum

                # 2. impulse along the normal
                vn = (b.velocity - a.velocity) @ normal
                if vn > 0:
                    continue  # already separating
                J = -(1.0 + self.restitution) * vn / inv_sum
                a.velocity -= J * normal * inv_a
                b.velocity += J * normal * inv_b

    def step(self, dt):
        # leapfrog: half kick, drift, recompute forces, half kick
        self.compute_forces()
        for b in self.bodies:
            b.velocity += 0.5 * dt * b.force / b.mass
            b.position += dt * b.velocity
        self.compute_forces()
        for b in self.bodies:
            b.velocity += 0.5 * dt * b.force / b.mass

        self.handle_collisions()