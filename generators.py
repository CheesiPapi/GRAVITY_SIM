import numpy as np

from bodies import Body

SUN_MASS = 10000.0
SUN_RADIUS = 10.0


def _make_sun():
    return Body(position=[0, 0, 0], velocity=[0, 0, 0],
                mass=SUN_MASS, radius=SUN_RADIUS, color=(1.0, 1.0, 0.0))


def _spaced(draw, placed, min_sep, tries=200):
    """Call draw() until it returns a position at least min_sep from every position in
    `placed` (gives up after `tries` and accepts the last draw). Random clumps of
    bodies start with violent close encounters; spacing them avoids that."""
    for _ in range(tries):
        candidate = draw()
        if all(np.linalg.norm(candidate - p) >= min_sep for p in placed):
            break
    placed.append(candidate)
    return candidate


def _radius_for(mass):
    """Radius of a body of this mass at constant density (so light bodies stay visible)."""
    return 0.6 * np.cbrt(mass)


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
    placed = []
    for _ in range(num_bodies):
        mass = np.random.uniform(1, 4)       # light, so the disk's own gravity stays gentle
        radius = _radius_for(mass)

        def draw():
            r = np.random.uniform(r_min, r_max)
            theta = np.random.uniform(0, 2 * np.pi)
            return np.array([r * np.cos(theta), r * np.sin(theta),
                             np.random.normal(0, thickness)])
        position = _spaced(draw, placed, min_sep=10.0)
        r = np.hypot(position[0], position[1])
        theta = np.arctan2(position[1], position[0])

        speed = np.sqrt(G * SUN_MASS / r)
        velocity = speed * np.array([-np.sin(theta), np.cos(theta), 0.0])
        velocity += np.random.normal(0, 0.02 * speed, size=3)   # small wobble

        bodies.append(Body(position, velocity, mass, radius))
    return bodies


# ---------------------------------------------------------------------------
# More scenarios
# ---------------------------------------------------------------------------

def _recentre(bodies):
    """Shift positions/velocities so the center of mass is at rest at the origin."""
    mass = np.array([b.mass for b in bodies])
    total = mass.sum()
    com = sum(b.mass * b.position for b in bodies) / total
    drift = sum(b.mass * b.velocity for b in bodies) / total
    for b in bodies:
        b.position -= com
        b.velocity -= drift
    return bodies


def _rotation_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def make_cloud(num_bodies, G, radius=100.0, spin=0.35):
    """Slowly spinning ball of similar-mass bodies, no sun. Collapses and clumps
    (best with 'collisions merge')."""
    masses = np.random.uniform(10, 30, num_bodies)
    total = masses.sum()
    omega = spin * np.sqrt(G * total / radius ** 3)
    v_circ = np.sqrt(G * total / radius)

    bodies = []
    for m in masses:
        direction = np.random.normal(size=3)
        direction /= np.linalg.norm(direction)
        position = direction * radius * np.cbrt(np.random.uniform())
        velocity = omega * np.cross([0.0, 0.0, 1.0], position)
        velocity += np.random.normal(0, 0.15 * v_circ, size=3)       # random motion
        shade = np.random.uniform(0.45, 0.8)
        bodies.append(Body(position, velocity, m, m * 0.05, (shade, shade, shade)))
    return _recentre(bodies)


def make_binary(num_planets, G, separation=40.0):
    """Two stars orbiting each other, with planets circling the pair."""
    m = 5000.0
    v_rel = np.sqrt(G * 2 * m / separation)               # circular relative speed
    stars = [
        Body([-separation / 2, 0, 0], [0, -v_rel / 2, 0], m, 6.0, (1.0, 0.85, 0.3)),
        Body([separation / 2, 0, 0], [0, v_rel / 2, 0], m, 6.0, (1.0, 0.5, 0.25)),
    ]
    bodies = list(stars)
    for _ in range(num_planets):
        mass = np.random.uniform(5, 20)
        r = np.random.uniform(120, 260)                   # well outside the pair
        theta = np.random.uniform(0, 2 * np.pi)
        tilt = np.random.normal(0, 0.08)
        position = [r * np.cos(theta), r * np.sin(theta), r * np.sin(tilt)]
        speed = np.sqrt(G * 2 * m / r)
        velocity = speed * np.array([-np.sin(theta), np.cos(theta), 0.0])
        bodies.append(Body(position, velocity, mass, _radius_for(mass)))
    return _recentre(bodies)


def make_figure8(num_bodies, G, mass=1000.0, size=50.0):
    """Three equal masses chasing each other round a figure-8 (Chenciner-Montgomery).
    The count is ignored. Any bump (or enough time) will eventually unravel it."""
    p = np.array([0.97000436, -0.24308753, 0.0])
    v3 = np.array([-0.93240737, -0.86473146, 0.0])
    v_scale = np.sqrt(G * mass / size)
    colors = [(1.0, 0.4, 0.4), (0.4, 1.0, 0.5), (0.5, 0.6, 1.0)]
    positions = [-p * size, p * size, np.zeros(3)]
    velocities = [-v3 / 2 * v_scale, -v3 / 2 * v_scale, v3 * v_scale]
    return [Body(pos, vel, mass, 6.0, col)
            for pos, vel, col in zip(positions, velocities, colors)]


def make_ring(num_bodies, G, r_min=24.0, r_max=48.0):
    """Big planet with a thin ring of tiny bodies."""
    planet_mass = 6000.0
    bodies = [Body([0, 0, 0], [0, 0, 0], planet_mass, 12.0, (0.9, 0.75, 0.5))]
    placed = []
    for _ in range(num_bodies):
        mass = np.random.uniform(0.05, 0.2)
        def draw():
            r = np.random.uniform(r_min, r_max)
            theta = np.random.uniform(0, 2 * np.pi)
            return np.array([r * np.cos(theta), r * np.sin(theta), np.random.normal(0, 0.4)])
        position = _spaced(draw, placed, min_sep=2.5)
        r = np.hypot(position[0], position[1])
        theta = np.arctan2(position[1], position[0])
        speed = np.sqrt(G * planet_mass / r)
        velocity = speed * np.array([-np.sin(theta), np.cos(theta), 0.0])
        shade = np.random.uniform(0.6, 0.95)
        bodies.append(Body(position, velocity, mass, 0.22, (shade, shade * 0.9, shade * 0.75)))
    return bodies


def _spin_disk(center, drift, tilt, central_mass, n, G, color, r_min=20.0, r_max=80.0):
    rot = _rotation_x(tilt)
    bodies = [Body(center, drift, central_mass, 8.0, color)]
    placed = []
    for _ in range(n):
        mass = np.random.uniform(2, 8)
        def draw():
            r = np.random.uniform(r_min, r_max)
            theta = np.random.uniform(0, 2 * np.pi)
            return np.array([r * np.cos(theta), r * np.sin(theta), np.random.normal(0, 2.0)])
        local_pos = _spaced(draw, placed, min_sep=8.0)
        r = np.hypot(local_pos[0], local_pos[1])
        theta = np.arctan2(local_pos[1], local_pos[0])
        speed = np.sqrt(G * central_mass / r)
        local_vel = speed * np.array([-np.sin(theta), np.cos(theta), 0.0])
        tint = tuple(0.35 + 0.65 * c for c in color)
        bodies.append(Body(rot @ local_pos + center, rot @ local_vel + drift,
                           mass, _radius_for(mass), tint))
    return bodies


def make_galaxies(num_bodies, G):
    """Two spinning disks (one tilted) on a collision course. num_bodies is the total."""
    n = max(1, num_bodies // 2)
    first = _spin_disk(np.array([-170.0, -50.0, 0.0]), np.array([45.0, 5.0, 0.0]),
                       0.0, 5000.0, n, G, (1.0, 0.7, 0.35))
    second = _spin_disk(np.array([170.0, 50.0, 0.0]), np.array([-45.0, -5.0, 0.0]),
                        np.radians(50), 5000.0, n, G, (0.4, 0.7, 1.0))
    return _recentre(first + second)


# name -> (builder, default count, description, minimum cull distance or None,
#          collision mode to switch to or None)
SCENARIOS = {
    "solar":    (make_solar_system, 5,   "sun with random tilted planet orbits", None, None),
    "disk":     (make_disk,         40,  "sun with a flat rotating disk (clumps over time)", None, None),
    "cloud":    (make_cloud,        60,  "spinning ball that collapses and clumps", None, "merge"),
    "binary":   (make_binary,       6,   "two stars with planets circling the pair", None, None),
    "figure8":  (make_figure8,      3,   "three equal bodies on a figure-8 orbit", None, None),
    "ring":     (make_ring,         150, "planet with a thin ring (uses bounce so it persists)", None, "bounce"),
    "galaxies": (make_galaxies,     80,  "two disks colliding (count = total)", 1500.0, None),
}