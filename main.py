import numpy as np
from bodies import Planet
from renderer import SimulationRenderer # Import our new tool!

def generate_bodies(num_bodies):
    bodies = []
    
    # Create the central Star manually
    sun = Planet(mass=10000.0, radius=8.0, position=[0,0,0], velocity=[0,0,0])
    bodies.append(sun)
    
    # Generate the random orbiting planets
    for _ in range(num_bodies):
        mass = np.random.uniform(10, 50)
        radius = mass * 0.05 
        position = np.random.uniform(-100, 100, size=3)
        velocity = np.random.uniform(-50, 50, size=3)
        bodies.append(Planet(mass, radius, position, velocity))
        
    return bodies
    

# --- RUN THE UNIVERSE ---
if __name__ == "__main__":
    print("Generating universe...")
    my_solar_system = generate_bodies(20)
    # Our time step (dt). If things move too fast/slow, change this number.
    dt = 0.01
    
    print("Launching simulator...")
    sim = SimulationRenderer(my_solar_system, dt)
    sim.start()