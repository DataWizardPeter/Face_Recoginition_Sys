import face_recognition
import os
import pickle
import re  # For cleaning names

# Define dataset paths
THREAT_PATH = "C:/Dataset/threat_faces"  # Folder containing images of threats
NON_THREAT_PATH = "C:/Dataset/non_threat_faces"  # Folder containing images of non-threats

# Initialize lists to store encodings and labels
face_encodings = []
face_labels = []

def clean_name(image_name):
    """
    Extracts the name from an image filename, removing numbers in parentheses.
    Example: 'peter(1).jpg' -> 'peter'
    """
    name = os.path.splitext(image_name)[0]  # Remove file extension
    name = re.sub(r'\(\d+\)', '', name)  # Remove numbers in parentheses
    return name.strip()  # Remove spaces if any

def encode_faces(folder_path):
    """
    Function to encode faces from images in a folder.

    Parameters:
        folder_path (str): Path to the folder containing face images.
    """
    if "non_threat" in folder_path.lower():
        label_type = "non-threat"
    elif "threat" in folder_path.lower():
        label_type = "threat"
    else:
        label_type = "unknown"  # Fallback if neither is found

    print(f"🔄 Processing images from: {folder_path}")

    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder '{folder_path}' not found!")
        return

    image_files = os.listdir(folder_path)
    if not image_files:
        print(f"⚠️ Warning: No images found in {folder_path}")
        return

    for image_name in image_files:
        image_path = os.path.join(folder_path, image_name)
        print(f"📷 Processing image: {image_name}")

        try:
            # Load image using face_recognition
            image = face_recognition.load_image_file(image_path)

            # Detect face and generate encodings
            encodings = face_recognition.face_encodings(image)

            if encodings:  # If at least one face is found
                face_encodings.append(encodings[0])  # Store the first face encoding
                label = f"{clean_name(image_name)} ({label_type})"
                face_labels.append(label)
                print(f"✅ Face encoded: {label}")
            else:
                print(f"⚠️ No face detected in {image_name}")

        except Exception as e:
            print(f"❌ Error processing {image_name}: {str(e)}")

# Encode faces from both folders
encode_faces(THREAT_PATH)
encode_faces(NON_THREAT_PATH)

# Save the trained model using pickle
model_data = {"encodings": face_encodings, "labels": face_labels}
model_filename = "face_model_v3.pkl"

try:
    with open(model_filename, "wb") as f:
        pickle.dump(model_data, f)
    print(f"✅ Face model trained and saved as '{model_filename}'!")
except Exception as e:
    print(f"❌ Error saving model: {str(e)}")