from console import Console
from generators import make_solar_system
from physics import Simulation
from renderer import SimulationRenderer

G = 200.0

if __name__ == "__main__":
    print("Generating universe...")
    bodies = make_solar_system(5, G)
    sim = Simulation(bodies, G=G, softening=0.1, restitution=1.0)

    print("Launching simulator...")
    app = SimulationRenderer(sim, dt=0.01)

    console = Console(app)
    app.attach_console(console)
    console.say("Press Enter to type a command ('help' lists them).")

    app.start()