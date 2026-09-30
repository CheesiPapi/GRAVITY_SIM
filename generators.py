import numpy as np

from bodies import Body

SUN_MASS = 10000.0
SUN_RADIUS = 10.0


def _make_sun():
    return Body(position=[0, 0, 0], velocity=[0, 0, 0],
                mass=SUN_MASS, radius=SUN_RADIUS, color=(1.0, 1.0, 0.0))


def make_solar_system(num_planets, G):
    """Central sun plus planets on randomly tilted, roughly circular orbits."""
    bodies = [_make_sun()]
    for _ in range(num_planets):
        mass = np.random.uniform(10, 50)
        radius = mass * 0.05

        direction = np.random.normal(size=3)
        direction /= np.linalg.norm(direction)
        r = np.random.uniform(40, 120)          # stay outside the sun
        position = direction * r

        # any vector perpendicular to the radius, scaled to circular-orbit speed
        tangent = np.cross(position, np.random.normal(size=3))
        tangent /= np.linalg.norm(tangent)
        velocity = tangent * np.sqrt(G * SUN_MASS / r)

        bodies.append(Body(position, velocity, mass, radius))
    return bodies


def make_disk(num_bodies, G, r_min=40.0, r_max=140.0, thickness=3.0):
    """Sun plus bodies in a thin disk, all orbiting the same way."""
    bodies = [_make_sun()]
    for _ in range(num_bodies):
        mass = np.random.uniform(10, 50)
        radius = mass * 0.05

        r = np.random.uniform(r_min, r_max)
        theta = np.random.uniform(0, 2 * np.pi)
        position = [r * np.cos(theta), r * np.sin(theta),
                    np.random.normal(0, thickness)]

        speed = np.sqrt(G * SUN_MASS / r)
        velocity = speed * np.array([-np.sin(theta), np.cos(theta), 0.0])
        velocity += np.random.normal(0, 0.02 * speed, size=3)   # small wobble

        bodies.append(Body(position, velocity, mass, radius))
    return bodies
