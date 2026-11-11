import trajectory
import marker

def main():
    drop_distance = trajectory.final_distance_yds
    print(f"Using drop distance: {drop_distance:.2f} yards for map display.")
    marker.show_marked_map(drop_distance)

if __name__ == "__main__":
    main()