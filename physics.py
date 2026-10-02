"""N-body gravity with two collision modes.

collision_mode = "bounce": bodies bounce elastically (restitution) and persist.
collision_mode = "merge":  colliding bodies merge, or shatter into fragments if
                           the impact energy exceeds their gravitational
                           binding energy. Mass and momentum are conserved.

The simulation knows nothing about VTK. Each merge/shatter appends an event
dict to self.events; the renderer collects them with pop_events() and draws a
flash. Bodies that disappear or appear are simply removed from / added to
self.bodies; the renderer keeps its actors in sync with that list.
"""
import numpy as np

from bodies import Body

MAX_RESOLUTIONS_PER_STEP = 50   # safety cap against endless collision cascades


def spread_directions(n):
    """n roughly evenly spaced unit vectors on a sphere, randomly rotated."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    dirs = np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)
    q, _ = np.linalg.qr(np.random.normal(size=(3, 3)))
    return dirs @ q.T


class Simulation:
    def __init__(self, bodies, G=200.0, softening=0.1, restitution=1.0,
                 collision_mode="merge", shatter_factor=1.5,
                 min_fragment_mass=2.0, cull_distance=500.0):
        self.bodies = bodies
        self.G = G
        self.softening = softening
        self.restitution = restitution
        self.collision_mode = collision_mode      # "merge" or "bounce"
        self.shatter_factor = shatter_factor      # higher = harder to shatter
        self.min_fragment_mass = min_fragment_mass
        self.cull_distance = cull_distance        # None = never delete far bodies
        self.events = []
        self.time = 0.0                           # simulated time elapsed
        self.step_count = 0

    # ---------- gravity ----------
    def compute_forces(self):
        for b in self.bodies:
            b.force[:] = 0.0
        n = len(self.bodies)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.bodies[i], self.bodies[j]
                diff = b.position - a.position
                dist2 = diff @ diff + self.softening ** 2
                dist = np.sqrt(dist2)
                f = self.G * a.mass * b.mass / dist2
                force = f * diff / dist
                a.force += force      # equal and opposite
                b.force -= force

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
        self._cull_distant()
        self.time += dt
        self.step_count += 1

    def _cull_distant(self):
        """Delete bodies farther than cull_distance from the center of mass."""
        if self.cull_distance is None or len(self.bodies) < 2:
            return
        pos = np.array([b.position for b in self.bodies])
        mass = np.array([b.mass for b in self.bodies])
        center = (mass[:, None] * pos).sum(axis=0) / mass.sum()
        keep = np.linalg.norm(pos - center, axis=1) <= self.cull_distance
        if not keep.all():
            self.bodies[:] = [b for b, k in zip(self.bodies, keep) if k]

    def pop_events(self):
        events, self.events = self.events, []
        return events

    # ---------- collisions ----------
    def handle_collisions(self):
        if self.collision_mode == "bounce":
            self._bounce_all()
            return
        for _ in range(MAX_RESOLUTIONS_PER_STEP):
            pair = self._find_overlap()
            if pair is None:
                return
            self._merge_or_shatter(*pair)

    def _find_overlap(self):
        n = len(self.bodies)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.bodies[i], self.bodies[j]
                if np.linalg.norm(b.position - a.position) < a.radius + b.radius:
                    return a, b
        return None

    # -- bounce mode --
    def _bounce_all(self):
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

    # -- merge / shatter mode --
    def binding_energy(self, body):
        return 0.6 * self.G * body.mass ** 2 / body.radius   # uniform sphere

    @staticmethod
    def collision_energy(a, b):
        mu = a.mass * b.mass / (a.mass + b.mass)              # reduced mass
        v_rel = np.linalg.norm(a.velocity - b.velocity)
        return 0.5 * mu * v_rel ** 2

    def _merge_or_shatter(self, a, b):
        total = a.mass + b.mass
        energy = self.collision_energy(a, b)
        threshold = self.shatter_factor * (self.binding_energy(a) + self.binding_energy(b))

        n = int(np.clip(3 + energy / threshold, 3, 12))
        n = min(n, int(total // self.min_fragment_mass))
        shatter = energy > threshold and n >= 2

        point = (a.position + b.position) / 2
        new_bodies = self._fragment(a, b, n) if shatter else [self._merge(a, b)]

        self.bodies.remove(a)
        self.bodies.remove(b)
        self.bodies.extend(new_bodies)
        self.events.append({"point": point, "size": a.radius + b.radius,
                            "shatter": shatter})

    @staticmethod
    def _merge(a, b):
        m = a.mass + b.mass
        position = (a.position * a.mass + b.position * b.mass) / m
        velocity = (a.velocity * a.mass + b.velocity * b.mass) / m
        radius = (a.radius ** 3 + b.radius ** 3) ** (1 / 3)   # volumes add
        color = a.color if a.mass >= b.mass else b.color
        return Body(position, velocity, m, radius, color)

    @staticmethod
    def _fragment(a, b, n):
        m = a.mass + b.mass
        com_pos = (a.position * a.mass + b.position * b.mass) / m
        com_vel = (a.velocity * a.mass + b.velocity * b.mass) / m
        v_rel = np.linalg.norm(a.velocity - b.velocity)
        kick_speed = 0.3 * v_rel
        color = a.color if a.mass >= b.mass else b.color

        weights = np.random.uniform(0.5, 1.5, n)
        masses = m * weights / weights.sum()

        dirs = spread_directions(n)
        kicks = dirs * kick_speed
        kicks -= (masses[:, None] * kicks).sum(axis=0) / m    # momentum-neutral

        density = m / (a.radius ** 3 + b.radius ** 3)         # preserves total volume
        radii = (masses / density) ** (1 / 3)
        # place fragments far enough out that they don't overlap each other
        spread = max(0.5 * (a.radius + b.radius), 2.2 * radii.max())

        return [Body(com_pos + d * spread, com_vel + k, mi, ri, color)
                for d, k, mi, ri in zip(dirs, kicks, masses, radii)]