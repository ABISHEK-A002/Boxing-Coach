import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Initialize variables
punch_count = 0
punch_start_time = None
prev_position = None
total_power = 0
speed = []

# Define heavy bag region (adjust these values based on your video)
HEAVY_BAG_REGION = {
    'x_min': 0.4,  # Adjust these normalized values (0 to 1) for the heavy bag area
    'x_max': 0.6,
    'y_min': 0.3,
    'y_max': 0.7
}

def is_in_heavy_bag_region(x, y):
    """Check if the given point (x, y) is inside the heavy bag region."""
    return HEAVY_BAG_REGION['x_min'] <= x <= HEAVY_BAG_REGION['x_max'] and HEAVY_BAG_REGION['y_min'] <= y <= HEAVY_BAG_REGION['y_max']

def calculate_speed_distance(pos1, pos2, time_elapsed):
    if pos1 is None or pos2 is None:
        return 0, 0
    distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
    speed = distance / time_elapsed if time_elapsed > 0 else 0
    return speed, distance

def analyze_boxing(video_path):
    global punch_count, punch_start_time, prev_position, total_power, speed

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        print("Error: Cannot open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Resize the frame to mobile screen dimensions
        frame = cv2.resize(frame, (480, 800))  # Typical mobile screen resolution

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Get wrist coordinates
            left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

            for wrist in [left_wrist, right_wrist]:
                wrist_pos = (wrist.x, wrist.y)

                # Check if wrist is in heavy bag region
                if is_in_heavy_bag_region(wrist.x, wrist.y):
                    spd, dist = calculate_speed_distance(prev_position, wrist_pos, 1/fps)
                    if dist > 0.02:  # Threshold for punch movement
                        punch_count += 1
                        speed.append(spd)
                        power = spd * 10  # Simplified power estimation
                        total_power += power

                prev_position = wrist_pos

        end_time = time.time()

        # Display the analyzed data
        cv2.putText(frame, f'Punch Count: {punch_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Average Power: {total_power/punch_count if punch_count > 0 else 0:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Average Speed: {np.mean(speed):.2f}', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Boxing Analysis', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Final Punch Count: {punch_count/10:.2f}")
    print(f"Average Power: {total_power/punch_count if punch_count > 0 else 0:.2f}")
    print(f"Average Speed: {np.mean(speed):.2f}")

# Replace 'boxing_video.mp4' with your video file path
analyze_boxing('box6.mp4')
