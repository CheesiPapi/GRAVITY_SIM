import numpy as np
from bodies import Body
from physics import Simulation
from renderer import SimulationRenderer

G = 200.0

def orbit_velocity(sun, position):
    """Velocity for a roughly circular orbit around the sun."""
    r_vec = position - sun.position
    r = np.linalg.norm(r_vec)
    speed = np.sqrt(G * sun.mass / r)
    tangent = np.cross(r_vec, np.random.normal(size=3))  # perpendicular to r_vec
    tangent /= np.linalg.norm(tangent)
    return tangent * speed

def generate_bodies(num_bodies):
    sun = Body(position=[0, 0, 0], velocity=[0, 0, 0],
               mass=10000.0, radius=10.0, color=(1.0, 1.0, 0.0))
    bodies = [sun]

    for _ in range(num_bodies):
        mass = np.random.uniform(10, 50)
        radius = mass * 0.05

        direction = np.random.normal(size=3)
        direction /= np.linalg.norm(direction)
        position = direction * np.random.uniform(40, 120)   # stay outside the sun

        velocity = orbit_velocity(sun, position)
        bodies.append(Body(position, velocity, mass, radius))

    return bodies

if __name__ == "__main__":
    print("Generating universe...")
    bodies = generate_bodies(5)
    sim = Simulation(bodies, G=G, softening=0.1, restitution=1.0)

    print("Launching simulator...")
    app = SimulationRenderer(sim, dt=0.01)
    app.start()