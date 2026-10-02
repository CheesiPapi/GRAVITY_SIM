"""Terminal-style command console for the gravity simulator.

A background thread reads lines from the terminal and puts them on a queue.
The VTK timer callback calls Console.process() every tick, which runs the
queued commands on the main thread (VTK is not thread-safe, so commands must
never touch VTK objects from the input thread).
"""
import queue
import threading

import numpy as np

import generators


def center_of_mass(bodies):
    if not bodies:
        return np.zeros(3)
    masses = np.array([b.mass for b in bodies])
    positions = np.array([b.position for b in bodies])
    return (masses[:, None] * positions).sum(axis=0) / masses.sum()


class Console:
    def __init__(self, app):
        self.app = app            # the SimulationRenderer
        self.follow = None        # None, "com", or a Body
        self._queue = queue.Queue()
        self._done = threading.Event()
        self._commands = {}       # name -> (handler, usage, description)
        self._register_commands()

    # ---------- setup ----------
    def add(self, names, handler, usage, description):
        """Register a command. `names` is 'name|alias|...'; may be two words."""
        for name in names.split("|"):
            self._commands[name] = (handler, usage, description)

    def _register_commands(self):
        self.add("help|?", self.cmd_help, "help", "show this list")
        self.add("speed up|faster", self.cmd_speed_up, "speed up [factor]", "run faster (default x2)")
        self.add("slow down|slower", self.cmd_slow_down, "slow down [factor]", "run slower (default /2)")
        self.add("speed", self.cmd_speed, "speed [value]", "show or set speed (1 = normal)")
        self.add("pause", self.cmd_pause, "pause", "freeze the simulation")
        self.add("resume|play", self.cmd_resume, "resume", "unfreeze the simulation")
        self.add("recenter", self.cmd_recenter, "recenter [com|<n>]", "aim camera at center of mass or body n")
        self.add("follow", self.cmd_follow, "follow [com|<n>|off]", "keep camera on center of mass or body n")
        self.add("data|positions", self.cmd_data, "data [n]", "mass/position/velocity of all bodies, or body n")
        self.add("collisions", self.cmd_collisions, "collisions [bounce|merge]", "bounce off, or merge/shatter")
        self.add("shatter", self.cmd_shatter, "shatter [factor]", "show/set shatter threshold (higher = harder)")
        self.add("disk", self.cmd_disk, "disk [count]", "restart as a rotating disk around a sun")
        self.add("solar", self.cmd_solar, "solar [count]", "restart with random orbiting planets")
        self.add("status", self.cmd_status, "status", "show speed, pause state, body count")
        self.add("quit|exit", self.cmd_quit, "quit", "close the simulator")

    def start(self):
        threading.Thread(target=self._read_loop, daemon=True).start()

    # ---------- input thread ----------
    def _read_loop(self):
        print("Console ready. Type 'help' for commands.")
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                return
            if not line:
                continue
            self._done.clear()
            self._queue.put(line)
            self._done.wait()     # don't re-prompt until the command has printed its output

    # ---------- main thread (called from the VTK timer) ----------
    def process(self):
        while True:
            try:
                line = self._queue.get_nowait()
            except queue.Empty:
                break
            self.execute(line)
            self._done.set()
        if self.follow is not None:
            bodies = self.app.sim.bodies
            if not isinstance(self.follow, str) and self.follow not in bodies:
                print("followed body no longer exists; following center of mass")
                self.follow = "com"
            self._look_at(self._target_point(self.follow))

    def execute(self, line):
        words = line.lower().split()
        for n in (2, 1):          # try a two-word command first ("speed up")
            name = " ".join(words[:n])
            if name in self._commands:
                try:
                    self._commands[name][0](words[n:])
                except Exception as exc:
                    print(f"error: {exc}")
                return
        print(f"unknown command: {line!r} (try 'help')")

    # ---------- helpers ----------
    def _target(self, arg):
        """'com' stays 'com'; a number becomes that Body (so merges don't shift it)."""
        if arg == "com":
            return "com"
        return self.app.sim.bodies[int(arg)]

    def _target_point(self, target):
        if isinstance(target, str):
            return center_of_mass(self.app.sim.bodies)
        return target.position.copy()

    def _look_at(self, point):
        camera = self.app.renderer.GetActiveCamera()
        focal = np.array(camera.GetFocalPoint())
        position = np.array(camera.GetPosition())
        delta = point - focal
        camera.SetFocalPoint(*point)
        camera.SetPosition(*(position + delta))
        self.app.renderer.ResetCameraClippingRange()

    def _set_speed(self, value):
        self.app.speed = float(np.clip(value, 1 / 32, 32))
        print(f"speed x{self.app.speed:g}")

    @staticmethod
    def _fmt(v):
        return "(" + ", ".join(f"{x:9.2f}" for x in v) + ")"

    # ---------- commands ----------
    def cmd_help(self, args):
        seen = set()
        for handler, usage, description in self._commands.values():
            if handler not in seen:
                seen.add(handler)
                print(f"  {usage:<26} {description}")

    def cmd_speed_up(self, args):
        self._set_speed(self.app.speed * (float(args[0]) if args else 2.0))

    def cmd_slow_down(self, args):
        self._set_speed(self.app.speed / (float(args[0]) if args else 2.0))

    def cmd_speed(self, args):
        if args:
            self._set_speed(float(args[0]))
        else:
            print(f"speed x{self.app.speed:g}")

    def cmd_pause(self, args):
        self.app.paused = True
        print("paused")

    def cmd_resume(self, args):
        self.app.paused = False
        print("running")

    def cmd_recenter(self, args):
        self._look_at(self._target_point(self._target(args[0] if args else "com")))
        self.app.render_window.Render()
        print("recentered")

    def cmd_follow(self, args):
        arg = args[0] if args else "com"
        if arg == "off":
            self.follow = None
            print("follow off")
            return
        self.follow = self._target(arg)
        print(f"following {arg}")

    def cmd_data(self, args):
        bodies = self.app.sim.bodies
        indices = [int(args[0])] if args else range(len(bodies))
        print(f"{'#':>3} {'mass':>9} {'radius':>7}  {'position':<34} velocity")
        for i in indices:
            b = bodies[i]
            print(f"{i:>3} {b.mass:>9.2f} {b.radius:>7.2f}  "
                  f"{self._fmt(b.position):<34} {self._fmt(b.velocity)}")
        if not args:
            print(f"center of mass: {self._fmt(center_of_mass(bodies))}")

    def cmd_collisions(self, args):
        sim = self.app.sim
        if args:
            if args[0] not in ("bounce", "merge"):
                print("usage: collisions [bounce|merge]")
                return
            sim.collision_mode = args[0]
        print(f"collisions: {sim.collision_mode}")

    def cmd_shatter(self, args):
        sim = self.app.sim
        if args:
            sim.shatter_factor = float(args[0])
        print(f"shatter factor: {sim.shatter_factor:g} (higher = harder to shatter)")

    def cmd_disk(self, args):
        count = int(args[0]) if args else 40
        self.follow = None
        self.app.set_bodies(generators.make_disk(count, self.app.sim.G))
        print(f"started disk with {count} bodies")

    def cmd_solar(self, args):
        count = int(args[0]) if args else 5
        self.follow = None
        self.app.set_bodies(generators.make_solar_system(count, self.app.sim.G))
        print(f"started solar system with {count} planets")

    def cmd_status(self, args):
        follow = "off" if self.follow is None else ("com" if isinstance(self.follow, str) else "a body")
        print(f"bodies: {len(self.app.sim.bodies)}  speed: x{self.app.speed:g}  "
              f"paused: {self.app.paused}  dt: {self.app.dt}  "
              f"collisions: {self.app.sim.collision_mode}  follow: {follow}")

    def cmd_quit(self, args):
        print("closing simulator")
        self.app.render_window.Finalize()
        self.app.interactor.TerminateApp()