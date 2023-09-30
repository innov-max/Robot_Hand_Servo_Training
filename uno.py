import cv2
import mediapipe as mp
import serial
import numpy as np

# Set up serial communication with Arduino
arduino = serial.Serial('/dev/ttyACM0', 9600)  # Replace '/dev/ttyACM0' with your Arduino's port

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Hands drawing module
mp_drawing = mp.solutions.drawing_utils

# Create a VideoCapture object to capture video from the webcam
cap = cv2.VideoCapture(0)

# Function to calculate angle between three points
def calculate_angle(point1, point2, point3):
    x1, y1, _ = point1.x, point1.y, point1.z
    x2, y2, _ = point2.x, point2.y, point2.z
    x3, y3, _ = point3.x, point3.y, point3.z

    radians = np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2)
    angle = np.abs(np.degrees(radians))

    return angle

while True:
    ret, frame = cap.read()  # Read a frame from the webcam

    # Convert the frame to RGB for use with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get landmarks of all fingers
            landmarks = hand_landmarks.landmark

            # Calculate angles between finger joints
            thumb_angle = calculate_angle(landmarks[3], landmarks[2], landmarks[1])
            index_angle = calculate_angle(landmarks[7], landmarks[6], landmarks[5])
            middle_angle = calculate_angle(landmarks[11], landmarks[10], landmarks[9])
            ring_angle = calculate_angle(landmarks[15], landmarks[14], landmarks[13])
            pinky_angle = calculate_angle(landmarks[19], landmarks[18], landmarks[17])

            # Trigger servo based on finger angles (customize the conditions as needed)
            if thumb_angle < 90 and index_angle < 90 and middle_angle < 90 and ring_angle < 90 and pinky_angle < 90:
                # Send a command to trigger the servo
                arduino.write(b'TRIGGER\n')

    # Draw landmarks on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture, close all OpenCV windows, and disconnect from the Arduino
cap.release()
cv2.destroyAllWindows()
arduino.close()
