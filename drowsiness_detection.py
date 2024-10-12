import streamlit as st
import cv2
import numpy as np
import dlib

# Load Dlib's face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[10])
    B = np.linalg.norm(mouth[4] - mouth[8])
    C = np.linalg.norm(mouth[0] - mouth[6])
    return (A + B) / (2.0 * C)

def detect_drowsiness_and_yawning():
    cap = cv2.VideoCapture(0)
    st.title("Drowsiness Detection")
    stframe = st.empty()  # Create a placeholder for the video frame

    # Render the stop button with a unique key
    if st.button("Stop Drowsiness Detection", key="stop_drowsiness_detection"):
        return  # Exit the function if the button is clicked

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
            mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)])

            # Calculate EAR and MAR
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
            mar = mouth_aspect_ratio(mouth)

            # Drowsiness Detection
            if ear < 0.25:  # Adjust this threshold
                drowsiness_status = "Drowsy"
                color = (0, 0, 255)  # Red color for drowsy
            else:
                drowsiness_status = "Awake"
                color = (0, 255, 0)  # Green color for alert

            # Display results on frame
            cv2.putText(frame, f"Drowsiness Status: {drowsiness_status}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.polylines(frame, [left_eye], isClosed=True, color=color, thickness=2)
            cv2.polylines(frame, [right_eye], isClosed=True, color=color, thickness=2)
            cv2.polylines(frame, [mouth], isClosed=True, color=color, thickness=2)

        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

    cap.release()

# Run the function in Streamlit
if __name__ == "__main__":
    detect_drowsiness_and_yawning()