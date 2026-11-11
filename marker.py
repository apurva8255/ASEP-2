import cv2
import numpy as np

def show_marked_map(drop_distance_yards):
    # === Load the golf course map ===
    img = cv2.imread("Golf Course Map Illustration.webp")
    print("Image Dimensions (Height, Width, Channels):", img.shape)

    start_x, start_y = 170, 200  # example coordinates for start point

    # === Define and Draw Scale (e.g., 100 yards) ===
    scale_yards = 100
    pixels_per_yard = 2  # calibrated to match map width ~50 yards ≈ 480 pixels
    scale_length_px = int(scale_yards * pixels_per_yard)
    scale_start = (50, img.shape[0] - 50)
    scale_end = (50 + scale_length_px, img.shape[0] - 50)

    marked_img = img.copy()
    # cv2.line(marked_img, scale_start, scale_end, (255, 255, 255), 2)
    cv2.putText(marked_img, f"{drop_distance_yards} yards", (scale_start[0], scale_start[1] - 10),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # === Define Drop Point Using Scale ===
    drop_x = start_x + int(drop_distance_yards / pixels_per_yard)
    drop_y = start_y

    # === Rotate trajectory by given angle (degrees) ===
    angle_deg = -45  # Example: 30° clockwise
    angle_rad = np.radians(angle_deg)

    dx = drop_x - start_x
    dy = drop_y - start_y

    rotated_dx = dx * np.cos(angle_rad) - dy * np.sin(angle_rad)
    rotated_dy = dx * np.sin(angle_rad) + dy * np.cos(angle_rad)

    drop_x = int(start_x + rotated_dx)
    drop_y = int(start_y + rotated_dy)

    # === Draw parabolic trajectory ===
    num_points = 50
    traj_points = []
    for i in range(num_points + 1):
        t = i / num_points
        x = int(start_x + t * (drop_x - start_x))
        y = int(start_y + t * (drop_y - start_y) - 100 * t * (1 - t))  # parabolic arc
        traj_points.append((x, y))

    pts = np.array(traj_points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(marked_img, [pts], isClosed=False, color=(255, 255, 0), thickness=3)

    # === Draw drop marker with reduced opacity ===
    overlay = marked_img.copy()
    cv2.circle(overlay, (drop_x, drop_y), 10, (0, 0, 255), -1)
    alpha = 0.4  # Opacity level
    cv2.addWeighted(overlay, alpha, marked_img, 1 - alpha, 0, marked_img)
    cv2.circle(marked_img, (start_x, start_y), 5, (0, 255, 0), -1)  # green filled circle

    # === Display and Save ===
    cv2.imshow("Drop Point on Course", marked_img)
    cv2.imwrite("marked_golf_course.png", marked_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    show_marked_map(172)