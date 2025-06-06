import joblib
import cv2
import numpy as np
from deepface import DeepFace
from sklearn.ensemble import RandomForestClassifier 
from datetime import datetime
import os
import pandas as pd
# Load the saved SVM model and label encoder
rf_classifier = joblib.load('/home/taharehan/Downloads/rkclassifier_2/rf_classifier.pkl')
label_encoder = joblib.load('/home/taharehan/Downloads/rkclassifier_2/label_encoder.pkl')


# Function to mark attendance in a CSV file
# Set to track individuals whose attendance has already been marked
attendance_set = set()

def mark_attendance(name):
    global attendance_set
    if name not in attendance_set:  # Check if attendance has already been marked
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        attendance_data = pd.DataFrame([[name, timestamp]], columns=["Name", "Timestamp"])
        attendance_file = "attendance.csv"

        # Append to CSV or create it if it doesn't exist
        if not os.path.exists(attendance_file):
            attendance_data.to_csv(attendance_file, index=False)
        else:
            attendance_data.to_csv(attendance_file, mode="a", header=False, index=False)

        attendance_set.add(name)  # Add the name to the set
        print(f"Attendance marked for {name} at {timestamp}")
    else:
        print(f"Attendance already marked for {name}")



# Real-time Face Recognition with webcam (using OpenCV)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    ).detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces_detected:
        face = frame[y : y + h, x : x + w]
        face = cv2.resize(face, (160, 160)) / 255.0  # Normalize the face

        # Extract face embedding using DeepFace
        result = DeepFace.represent(
            frame, model_name="VGG-Face", enforce_detection=False
        )
        face_embedding = result[0]["embedding"]

        # Flatten the embedding
        face_embedding = np.array(face_embedding).reshape(1, -1)

        # Recognize the face using the Random Forest classifier
        prediction = rf_classifier.predict(face_embedding)
        predicted_label = label_encoder.inverse_transform(prediction)

        # Mark attendance
        mark_attendance(predicted_label[0])

        # Draw a rectangle around the face and label it
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{predicted_label[0]}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Real-time Face Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
