import numpy as np
import matplotlib.pyplot as plt
import math

# === Club Data and Random Parameter Generation ===
import random

# Club-wise realistic ranges: (angle_min, angle_max, speed_min, speed_max)
club_data = {
    "D": (9, 15, 50, 75), #Driver
    "d": (9, 15, 50, 75),
    "3": (11, 16, 45, 70), #3Iron
    "5": (15, 21, 40, 60), #5Iron
    "7": (18, 25, 35, 50), #7Iron
    "9": (22, 30, 30, 45), #9Iron
    "P": (26, 35, 25, 40), #Pitching Wedge
    "p": (26, 35, 25, 40),
    "S": (28, 40, 20, 35), #Sand Wedge
    "s": (28, 40, 20, 35),
    "L": (30, 50, 15, 30), #Lob Wedge
    "l": (30, 50, 15, 30),
}

def generate_random_params(club_type):
    if club_type not in club_data:
        raise ValueError(f"Unknown club type: {club_type}")
    angle_min, angle_max, speed_min, speed_max = club_data[club_type]
    angle = random.uniform(angle_min, angle_max)
    speed = random.uniform(speed_min, speed_max)
    return speed, angle



club = input("Club Type: ")
rand_speed, rand_angle = generate_random_params(club)
print(f"\nRandom parameters for {club}:")
print(f"  Velocity: {rand_speed:.2f} m/s")
print(f"  Angle: {rand_angle:.2f}°")

# ==== Parameters ====
initial_speed = rand_speed  # m/s
launch_angle_deg = rand_angle  # degrees
g = 9.81  # m/s^2
dt = 0.05  # time interval (s)

# ==== Derived Components ====
angle_rad = np.radians(launch_angle_deg)
vx = initial_speed * np.cos(angle_rad)
vy = initial_speed * np.sin(angle_rad)

# ==== Time of Flight ====
time_of_flight = (2 * vy) / g
t_values = np.arange(0, time_of_flight + dt, dt)

# ==== Lists to Store Trajectory ====
x_values, y_values = [], []
x_yards = []

print(f"{'Time (s)':>8} | {'X (m)':>8} | {'Y (m)':>8} | {'Speed (m/s)':>12} | {'Angle (deg)':>12} | {'X (yd)':>10}")
print("-" * 72)

for t in t_values:
    x = vx * t
    y = vy * t - 0.5 * g * t**2
    if y < 0:
        break

    # Instantaneous velocity components
    vy_t = vy - g * t
    v_t = math.sqrt(vx**2 + vy_t**2)
    angle_t = math.degrees(math.atan2(vy_t, vx))

    print(f"{t:8.2f} | {x:8.2f} | {y:8.2f} | {v_t:12.2f} | {angle_t:12.2f} | {x * 1.09361:10.2f} yd")

    x_values.append(x)
    y_values.append(y)
    x_yards.append(x * 1.09361)  # Convert meters to yards

# After the loop: get the final horizontal distance in yards and print it
final_distance_yds = x_yards[-1]
print(f"\nFinal horizontal distance: {final_distance_yds:.2f} yards")



if __name__ == "__main__":
    plt.figure(figsize=(10, 5))
    plt.plot(x_yards, y_values, 'g--', label='Trajectory (Yards)')
    plt.title('2D Projectile Trajectory in Yards')
    plt.xlabel('Distance (yd)')
    plt.ylabel('Height (m)')
    plt.grid(True)
    plt.legend()
    plt.show()
