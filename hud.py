"""On-screen text: status HUD (top-left) and console log + prompt (bottom-left)."""
import vtk

HINT = "Enter: command   arrows: rotate   space: pause   v: vectors   t: trails   h: HUD"


def _make_text(size, color, opacity=0.4):
    actor = vtk.vtkTextActor()
    prop = actor.GetTextProperty()
    prop.SetFontFamilyToCourier()          # monospace so the columns line up
    prop.SetFontSize(size)
    prop.SetColor(*color)
    prop.SetBackgroundColor(0.0, 0.0, 0.0)
    prop.SetBackgroundOpacity(opacity)
    return actor


class Overlay:
    def __init__(self, app):
        self.app = app
        self.console = None            # set by SimulationRenderer.attach_console
        self.hud_visible = True
        self._frame = 0
        self._last = {}                # last text sent to each actor (skip redundant updates)

        self.hud = _make_text(14, (0.85, 0.95, 1.0))
        self.hud.GetTextProperty().SetVerticalJustificationToTop()
        self.hud.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
        self.hud.SetPosition(0.01, 0.985)

        self.log = _make_text(14, (0.8, 1.0, 0.8))
        self.log.GetTextProperty().SetVerticalJustificationToBottom()
        self.log.SetDisplayPosition(12, 44)

        self.prompt = _make_text(16, (1.0, 1.0, 1.0))
        self.prompt.GetTextProperty().SetVerticalJustificationToBottom()
        self.prompt.SetDisplayPosition(12, 12)

        for actor in (self.hud, self.log, self.prompt):
            app.renderer.AddViewProp(actor)

    def _set(self, name, actor, text):
        if self._last.get(name) != text:
            self._last[name] = text
            actor.SetInput(text if text else " ")
            actor.SetVisibility(bool(text))

    def update(self):
        """Called once per tick; only touches an actor when its text changed."""
        self._frame += 1
        console = self.console

        if console is not None:
            self._set("log", self.log, "\n".join(console.visible_log()))
            self._set("prompt", self.prompt,
                      ("> " + console.buffer + "_") if console.active else HINT)
        else:
            self._set("prompt", self.prompt, HINT)

        if self.hud_visible:
            if self._frame % 6 == 1:               # ~10 updates per second is plenty
                self._set("hud", self.hud, self._hud_text())
        self.hud.SetVisibility(self.hud_visible and bool(self._last.get("hud")))

    def _hud_text(self):
        app = self.app
        sim = app.sim
        bodies = sim.bodies

        mass = sum(b.mass for b in bodies)
        kinetic = sum(0.5 * b.mass * float(b.velocity @ b.velocity) for b in bodies)
        cull = "off" if sim.cull_distance is None else f"{sim.cull_distance:g}"
        follow = self.console.follow_label() if self.console else "off"
        state = "PAUSED" if app.paused else "RUNNING"
        if app.vector_mode == "off":
            vectors = "off"
        else:
            vectors = f"{app.vector_mode} x{app.vector_scale:g}  (cyan=vel red=acc)"

        rows = [
            f"time        {sim.time:<10.2f}  step {sim.step_count}",
            f"bodies      {len(bodies):<10d}  mass {mass:,.0f}",
            f"speed       x{app.speed:<9g}  {state}",
            f"collisions  {sim.collision_mode}",
            f"cull dist   {cull}",
            f"vectors     {vectors}",
            f"trails      {'off' if not app.trails.enabled else str(app.trails.length) + ' pts'}",
            f"follow      {follow}",
            f"energy (KE) {kinetic:,.0f}",
            f"fps {app.fps:4.0f}   physics {app.physics_ms:5.1f} ms",
        ]
        return "\n".join(rows)