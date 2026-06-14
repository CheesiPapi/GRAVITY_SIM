# physics file

import numpy as np

def update_physics(bodies, dt, G=6.67430e-11):
    # loop 1: pick a planet to update
    for planet_a in bodies:
        
        #reset planet a's total force to zero at the start of the frame
        total_force = np.array([0.0, 0.0, 0.0])
        
        # Loop 2: look at every other planet to see how hard it pulls on planet a
        for planet_b in bodies:
            #the golden rule: dont calculate gravity against yourself
            if planet_a is planet_b:
                continue
            
            diff = planet_b.position - planet_a.position
            dist = np.sqrt(np.sum(diff**2))
            
            if dist == 0:
                continue

            # calculate the gravity force
            F =G * ((planet_a.mass * planet_b.mass) / dist**2)

            unit_direction = diff / dist
            force_vector = F * unit_direction

            #add this planet's pull to our grand totol
            total_force += force_vector
        # now that we felt the pull of all planets, move planet a
        acceleration = total_force / planet_a.mass
        planet_a.velocity += acceleration * dt
        planet_a.position += planet_a.velocity * dt