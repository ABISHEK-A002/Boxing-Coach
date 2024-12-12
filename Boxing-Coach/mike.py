import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize text-to-speech engine
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

def calculate_angle(a, b, c):
    """Calculate the angle between three points"""
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # Endpoint
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def classify_move(pose_landmarks):
    """Classify the movement based on detected key points"""
    moves = {
        1: "Left hook to head",
        2: "Straight right to head",
        3: "Left uppercut",
        4: "Right uppercut",
        5: "Left hook to body",
        6: "Right hook to body",
        7: "Jab to head",
        8: "Jab to body",
    }

    # Extract key points
    left_shoulder = [pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                     pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
    left_elbow = [pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                  pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y]
    left_wrist = [pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                  pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y]

    right_shoulder = [pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                      pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
    right_elbow = [pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                   pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
    right_wrist = [pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                   pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]

    # Example detection for a left hook to head
    angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    if 70 <= angle <= 110 and left_wrist[0] > left_elbow[0]:
        return 1

    # Example detection for a straight right to head
    angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    if 160 <= angle <= 180 and right_wrist[1] < right_elbow[1]:
        return 2

    # Detection for left uppercut
    if left_wrist[1] > left_elbow[1] and left_wrist[0] < left_elbow[0]:
        return 3

    # Detection for right uppercut
    if right_wrist[1] > right_elbow[1] and right_wrist[0] > right_elbow[0]:
        return 4

    # Detection for left hook to body
    if 70 <= angle <= 110 and left_wrist[1] > left_elbow[1]:
        return 5

    # Detection for right hook to body
    if 70 <= angle <= 110 and right_wrist[1] > right_elbow[1]:
        return 6

    # Detection for jab to head (left jab)
    angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    if 160 <= angle <= 180 and left_wrist[1] < left_elbow[1]:
        return 7

    # Detection for jab to body (left jab to body)
    if 160 <= angle <= 180 and left_wrist[1] > left_elbow[1]:
        return 8

    return None

# Moves description
moves = {
    1: "Left hook to head",
    2: "Straight right to head",
    3: "Left uppercut",
    4: "Right uppercut",
    5: "Left hook to body",
    6: "Right hook to body",
    7: "Jab to head",
    8: "Jab to body",
}

# Session phases
def start_session():
    speak("Welcome to the boxing training session. Let's begin.")

    # Phase 1: Read moves aloud for 1 minute
    start_time = time.time()
    while time.time() - start_time < 60:
        for move_id, move_name in moves.items():
            speak(f"Move {move_id}: {move_name}")

    # Phase 2: Teach the moves step-by-step
    speak("Now, let's learn the moves step by step.")
    for move_id, move_name in moves.items():
        speak(f"Let's practice {move_name}. Follow my instructions.")
        time.sleep(15)  # Allow user time to practice each move

    # Phase 3: Evaluate user movements for 2 minutes
    cap = cv2.VideoCapture(0)
    speak("Now I will evaluate your movements.")
    evaluation_start_time = time.time()

    while cap.isOpened() and time.time() - evaluation_start_time < 120:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            move = classify_move(results.pose_landmarks.landmark)
            if move:
                speak(f"Great job! You did {moves[move]}.")
            else:
                speak("Try again. Make sure your form is correct.")

        # Display frame
        cv2.imshow("Boxing Evaluation", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    speak("Session completed. Great work!")

# Start the training session
start_session()
