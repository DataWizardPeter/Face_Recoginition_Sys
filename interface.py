import face_recognition
import cv2
import pickle
import numpy as np
import winsound  # Only works on Windows for playing alarm sound

# Load the trained face model from the saved file
with open("face_model_v3.pkl", "rb") as f:
    model_data = pickle.load(f)


# Extract stored encodings and labels
known_encodings = model_data["encodings"]
known_labels = model_data["labels"]

# Start webcam for real-time face recognition
video_capture = cv2.VideoCapture(0)

while True:
    # Capture a frame from the webcam
    ret, frame = video_capture.read()

    # Convert the frame from BGR (OpenCV format) to RGB (face_recognition format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face locations in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)

    # Generate encodings for detected faces
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each detected face
    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compute face distances and find the best match
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_idx = np.argmin(distances)  # Get index of closest match
        name = "Unknown"  # Default label
        label_type = "unknown"

        # Check if the match is reliable
        if distances[best_match_idx] < 0.6:  # Threshold for accuracy
            name_with_label = known_labels[best_match_idx]  # Retrieve the corresponding label
            name, label_type = name_with_label.rsplit(" ", 1)  # Separate name and label
            label_type = label_type.strip("()")  # Remove parentheses

            # If the detected face is labeled as "threat," trigger an alarm
            if label_type.lower() == "threat":
                print("🚨 THREAT DETECTED! 🚨")
                winsound.Beep(1000, 500)  # Play a beep sound (Only works on Windows)

        # Extract face coordinates for drawing a rectangle
        top, right, bottom, left = face_location

        # Set rectangle color: Red for "threat", Green for "non-threat", Blue for "unknown"
        if label_type.lower() == "threat":
            color = (0, 0, 255)  # Red
        elif label_type.lower() == "non-threat":
            color = (0, 255, 0)  # Green
        else:
            color = (255, 0, 0)  # Blue

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Display the detected name and label above the rectangle
        cv2.putText(frame, f"{name} ({label_type})", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show the live video with face recognition
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
video_capture.release()
cv2.destroyAllWindows()