"""Command console that lives inside the VTK window.

Keys arrive from SimulationRenderer.on_key via handle_key(). Output goes to
self.log (drawn on screen by hud.Overlay) and is also printed to the terminal.
Everything runs on the VTK thread, so there is no threading.
"""
import time
from collections import deque

import numpy as np

import generators

VECTOR_MODES = ("off", "vel", "acc", "both")

# keys that act while the console is closed (keysym -> command line)
HOTKEYS = {
    "space": "pause",
    "h": "hud",
    "v": "vectors",
    "c": "recenter",
    "plus": "speed up",
    "equal": "speed up",
    "minus": "slow down",
}

# keysym names for characters, used if VTK doesn't supply a key code
SYMBOL_KEYS = {"space": " ", "minus": "-", "period": ".", "comma": ",",
               "slash": "/", "underscore": "_", "plus": "+", "equal": "="}

LOG_SECONDS = 8.0       # how long output stays visible when the console is closed
LOG_LINES_CLOSED = 8
LOG_LINES_OPEN = 20
DATA_ROWS_ON_SCREEN = 10


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
        self.active = False       # True while the user is typing
        self.buffer = ""
        self.log = deque(maxlen=200)      # (timestamp, text)
        self.history = []
        self._hist_pos = 0
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
        self.add("pause", self.cmd_pause, "pause", "pause / unpause")
        self.add("resume|play", self.cmd_resume, "resume", "unpause")
        self.add("recenter", self.cmd_recenter, "recenter [com|<n>]", "aim camera at center of mass or body n")
        self.add("follow", self.cmd_follow, "follow [com|<n>|off]", "keep camera on center of mass or body n")
        self.add("vectors", self.cmd_vectors, "vectors [off|vel|acc|both]", "arrows (no arg = cycle)")
        self.add("hud", self.cmd_hud, "hud [on|off]", "show/hide the status panel")
        self.add("data|positions", self.cmd_data, "data [n]", "mass/position/velocity of bodies")
        self.add("collisions", self.cmd_collisions, "collisions [bounce|merge]", "bounce off, or merge/shatter")
        self.add("shatter", self.cmd_shatter, "shatter [factor]", "shatter threshold (higher = harder)")
        self.add("cull", self.cmd_cull, "cull [distance|off]", "delete bodies this far from the center")
        self.add("disk", self.cmd_disk, "disk [count]", "restart as a rotating disk around a sun")
        self.add("cloud", self.cmd_cloud, "cloud [count]", "restart with a cloud of bodies falling towards a center of gravity")
        self.add("solar", self.cmd_solar, "solar [count]", "restart with random orbiting planets")
        self.add("status", self.cmd_status, "status", "one-line status")
        self.add("quit|exit", self.cmd_quit, "quit", "close the simulator")

    # ---------- output ----------
    def say(self, text, screen=True):
        print(text)
        if screen:
            self.log.append((time.monotonic(), text))

    def visible_log(self):
        if self.active:
            return [t for _, t in list(self.log)[-LOG_LINES_OPEN:]]
        now = time.monotonic()
        fresh = [t for ts, t in self.log if now - ts < LOG_SECONDS]
        return fresh[-LOG_LINES_CLOSED:]

    # ---------- keyboard ----------
    @staticmethod
    def _printable(keysym, keycode):
        if isinstance(keycode, int):
            keycode = chr(keycode)
        if keycode and len(keycode) == 1 and 32 <= ord(keycode) < 127:
            return keycode
        if keysym in SYMBOL_KEYS:
            return SYMBOL_KEYS[keysym]
        if len(keysym) == 1:
            return keysym
        return ""

    def handle_key(self, keysym, keycode):
        """Returns True if the key was consumed (so the camera should ignore it)."""
        if not self.active:
            if keysym in ("Return", "KP_Enter", "slash"):
                self.active = True
                self.buffer = ""
                self._hist_pos = len(self.history)
                return True
            command = HOTKEYS.get(keysym.lower())
            if command:
                self.execute(command)
                return True
            return False

        if keysym in ("Return", "KP_Enter"):
            line = self.buffer.strip()
            self.buffer = ""
            if not line:                       # Enter on an empty line closes the console
                self.active = False
                return True
            self.history.append(line)
            self._hist_pos = len(self.history)
            self.say("> " + line)
            self.execute(line)
        elif keysym == "Escape":
            self.active = False
            self.buffer = ""
        elif keysym == "BackSpace":
            self.buffer = self.buffer[:-1]
        elif keysym == "Up":
            if self.history:
                self._hist_pos = max(0, self._hist_pos - 1)
                self.buffer = self.history[self._hist_pos]
        elif keysym == "Down":
            if self._hist_pos < len(self.history) - 1:
                self._hist_pos += 1
                self.buffer = self.history[self._hist_pos]
            else:
                self._hist_pos = len(self.history)
                self.buffer = ""
        elif keysym == "Tab":
            self._complete()
        else:
            self.buffer += self._printable(keysym, keycode)
        return True

    def _complete(self):
        prefix = self.buffer.lower()
        names = sorted(n for n in self._commands if n.startswith(prefix))
        if len(names) == 1:
            self.buffer = names[0] + " "
        elif names:
            self.say("  ".join(names))

    # ---------- per-tick work (called from the VTK timer) ----------
    def tick(self):
        if self.follow is None:
            return
        if not isinstance(self.follow, str) and self.follow not in self.app.sim.bodies:
            self.say("followed body no longer exists; following center of mass")
            self.follow = "com"
        self._look_at(self._target_point(self.follow))

    def follow_label(self):
        if self.follow is None:
            return "off"
        if isinstance(self.follow, str):
            return "center of mass"
        try:
            return f"body {self.app.sim.bodies.index(self.follow)}"
        except ValueError:
            return "lost"

    def execute(self, line):
        words = line.lower().split()
        for n in (2, 1):          # try a two-word command first ("speed up")
            name = " ".join(words[:n])
            if name in self._commands:
                try:
                    self._commands[name][0](words[n:])
                except Exception as exc:
                    self.say(f"error: {exc}")
                return
        self.say(f"unknown command: {line!r} (Tab completes, 'help' lists)")

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
        self.say(f"speed x{self.app.speed:g}")

    @staticmethod
    def _fmt(v):
        return "(" + ", ".join(f"{x:8.1f}" for x in v) + ")"

    # ---------- commands ----------
    def cmd_help(self, args):
        seen = set()
        for handler, usage, description in self._commands.values():
            if handler not in seen:
                seen.add(handler)
                self.say(f"{usage:<27}{description}")
        self.say("keys: Enter open/close  Esc close  Tab complete  Up/Down history")

    def cmd_speed_up(self, args):
        self._set_speed(self.app.speed * (float(args[0]) if args else 2.0))

    def cmd_slow_down(self, args):
        self._set_speed(self.app.speed / (float(args[0]) if args else 2.0))

    def cmd_speed(self, args):
        if args:
            self._set_speed(float(args[0]))
        else:
            self.say(f"speed x{self.app.speed:g}")

    def cmd_pause(self, args):
        self.app.paused = not self.app.paused
        self.say("paused" if self.app.paused else "running")

    def cmd_resume(self, args):
        self.app.paused = False
        self.say("running")

    def cmd_recenter(self, args):
        self._look_at(self._target_point(self._target(args[0] if args else "com")))
        self.say("recentered")

    def cmd_follow(self, args):
        arg = args[0] if args else "com"
        if arg == "off":
            self.follow = None
            self.say("follow off")
            return
        self.follow = self._target(arg)
        self.say(f"following {arg}")

    def cmd_vectors(self, args):
        app = self.app
        if not args:
            app.vector_mode = VECTOR_MODES[(VECTOR_MODES.index(app.vector_mode) + 1) % len(VECTOR_MODES)]
        elif args[0] == "scale":
            if len(args) < 2:
                self.say("usage: vectors scale <factor>")
                return
            app.vector_scale = float(args[1])
        elif args[0] in VECTOR_MODES:
            app.vector_mode = args[0]
        else:
            self.say("usage: vectors [off|vel|acc|both] | vectors scale <factor>")
            return
        self.say(f"vectors: {app.vector_mode}  scale x{app.vector_scale:g}")

    def cmd_hud(self, args):
        overlay = self.app.overlay
        if args and args[0] in ("on", "off"):
            overlay.hud_visible = (args[0] == "on")
        else:
            overlay.hud_visible = not overlay.hud_visible
        self.say(f"hud {'on' if overlay.hud_visible else 'off'}")

    def cmd_data(self, args):
        bodies = self.app.sim.bodies
        indices = [int(args[0])] if args else range(len(bodies))
        self.say(f"{'#':>3} {'mass':>8} {'rad':>5}  {'position':<30} velocity")
        for k, i in enumerate(indices):
            b = bodies[i]
            self.say(f"{i:>3} {b.mass:>8.1f} {b.radius:>5.1f}  "
                     f"{self._fmt(b.position):<30} {self._fmt(b.velocity)}",
                     screen=k < DATA_ROWS_ON_SCREEN)
        if len(indices) > DATA_ROWS_ON_SCREEN:
            self.say(f"... {len(indices) - DATA_ROWS_ON_SCREEN} more rows (full table in terminal)")
        if not args:
            self.say(f"center of mass: {self._fmt(center_of_mass(bodies))}")

    def cmd_collisions(self, args):
        sim = self.app.sim
        if args:
            if args[0] not in ("bounce", "merge"):
                self.say("usage: collisions [bounce|merge]")
                return
            sim.collision_mode = args[0]
        self.say(f"collisions: {sim.collision_mode}")

    def cmd_shatter(self, args):
        sim = self.app.sim
        if args:
            sim.shatter_factor = float(args[0])
        self.say(f"shatter factor: {sim.shatter_factor:g} (higher = harder to shatter)")

    def cmd_cull(self, args):
        sim = self.app.sim
        if args:
            sim.cull_distance = None if args[0] == "off" else float(args[0])
        value = "off" if sim.cull_distance is None else f"{sim.cull_distance:g}"
        self.say(f"cull distance: {value}")

    def cmd_disk(self, args):
        count = int(args[0]) if args else 40
        self.follow = None
        self.app.set_bodies(generators.make_disk(count, self.app.sim.G))
        self.say(f"started disk with {count} bodies")

    def cmd_solar(self, args):
        count = int(args[0]) if args else 5
        self.follow = None
        self.app.set_bodies(generators.make_solar_system(count, self.app.sim.G))
        self.say(f"started solar system with {count} planets")

    def cmd_cloud(self, args):
        count = int(args[0]) if args else 5
        self.follow = None
        self.app.set_bodies(generators.make_cloud_system(count, self.app.sim.G))
        self.say(f"started cloud system with {count} bodies")

    def cmd_status(self, args):
        a, s = self.app, self.app.sim
        self.say(f"bodies {len(s.bodies)}  speed x{a.speed:g}  paused {a.paused}  "
                 f"collisions {s.collision_mode}  follow {self.follow_label()}")

    def cmd_quit(self, args):
        self.say("closing simulator")
        self.app.render_window.Finalize()
        self.app.interactor.TerminateApp()