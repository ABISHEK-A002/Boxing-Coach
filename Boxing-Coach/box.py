import cv2
import mediapipe as mp
import time
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Start video capture
cap = cv2.VideoCapture(0)

# Function to capture hand markers
def capture_hand_markers(duration):
    print(f"Please show both hands for {duration} seconds.")
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Draw the markers for hands
            left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_x, left_y = int(left_wrist.x * frame.shape[1]), int(left_wrist.y * frame.shape[0])
            right_x, right_y = int(right_wrist.x * frame.shape[1]), int(right_wrist.y * frame.shape[0])

            cv2.circle(frame, (left_x, left_y), 10, (255, 0, 0), -1)  # Draw left hand marker
            cv2.circle(frame, (right_x, right_y), 10, (255, 0, 0), -1)  # Draw right hand marker

        cv2.imshow('Capture Hand Markers', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Session manually terminated.")
            break

# Function to capture bag marker
def capture_bag_marker(duration):
    print(f"Please show the bag for {duration} seconds.")
    start_time = time.time()
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Draw a marker for the bag
            bag_x, bag_y = int(frame.shape[1] * 0.8), int(frame.shape[0] * 0.5)  # Initial bag position
            cv2.circle(frame, (bag_x, bag_y), 10, (0, 0, 255), -1)  # Draw bag marker

        cv2.imshow('Capture Bag Marker', frame)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            print("Session manually terminated.")
            break

# Capture hand markers for 10 seconds
capture_hand_markers(10)

# Capture bag marker for 3 seconds
capture_bag_marker(3)

# Start main tracking
prev_x, prev_y = None, None
prev_time = time.time()
start_time = time.time()
session_duration = 120  # 2 minutes in seconds
interval_duration = 30  # 30-second intervals

# Storage for report data
interval_data = []
current_interval_start = start_time
interval_speeds = []
interval_impacts = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Calculate elapsed session time
    elapsed_time = time.time() - start_time
    if elapsed_time > session_duration:
        print("Session completed.")
        break
    
    # Convert to RGB for pose detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Update bag position based on detection
    if results.pose_landmarks:
        # Get wrist position
        wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        cur_x, cur_y = wrist.x, wrist.y
        cur_x, cur_y = int(cur_x * frame.shape[1]), int(cur_y * frame.shape[0])  # Convert to pixel values
        
        # Mark hand position on the frame
        cv2.circle(frame, (cur_x, cur_y), 10, (255, 0, 0), -1)  # Draw the hand marker
        
        # Update bag position for movement (example logic)
        bag_x, bag_y = cur_x + 200, cur_y  # Move the bag marker relative to the hand's position
        cv2.circle(frame, (bag_x, bag_y), 10, (0, 0, 255), -1)  # Draw bag marker
        
        cur_time = time.time()
        
        # Calculate punch speed and impact
        if prev_x is not None:
            distance = np.sqrt((cur_x - prev_x)**2 + (cur_y - prev_y)**2)
            time_elapsed = cur_time - prev_time
            speed = distance / time_elapsed
            impact = speed * 10  # Hypothetical impact formula

            # Store data for the current interval
            interval_speeds.append(speed)
            interval_impacts.append(impact)

            # Display results on frame
            cv2.putText(frame, f'Speed: {speed:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f'Impact: {impact:.2f}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Update previous values
        prev_x, prev_y = cur_x, cur_y
        prev_time = cur_time
    else:
        # Display warning if hand is not detected
        cv2.putText(frame, "Warning: Hand not in frame!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Check for 30-second intervals
    if time.time() - current_interval_start >= interval_duration:
        # Calculate average speed and impact for the interval
        avg_speed = np.mean(interval_speeds) if interval_speeds else 0
        avg_impact = np.mean(interval_impacts) if interval_impacts else 0
        interval_data.append((avg_speed, avg_impact))
        
        # Reset for the next interval
        interval_speeds = []
        interval_impacts = []
        current_interval_start = time.time()

    # Show the frame
    cv2.imshow('Boxing Trainer', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Session manually terminated.")
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()

# Generate session report
print("Training session report:")
for i, (avg_speed, avg_impact) in enumerate(interval_data, start=1):
    print(f"Interval {i} - Avg Speed: {avg_speed:.2f}, Avg Impact: {avg_impact:.2f}")

print("Training session finished and data processing completed.")
