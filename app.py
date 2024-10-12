import streamlit as st
import cv2
import numpy as np
from drowsiness_detection import detect_drowsiness_and_yawning

st.title("Driver Drowsiness and Yawning Detection")

if st.button("Start Detection"):
    cap = cv2.VideoCapture(0)
    frame_placeholder = st.empty()  # Placeholder for video frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Call the modified function without passing an argument
        detected_frame = detect_drowsiness_and_yawning()

        # Convert the image from BGR to RGB
        detected_frame = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        frame_placeholder.image(detected_frame, channels="RGB")

    cap.release()
