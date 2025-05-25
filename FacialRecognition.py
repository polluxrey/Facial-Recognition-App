import cv2
import os
import time
import streamlit as st
from deepface import DeepFace
from PIL import Image

# Constants
DB_PATH = "db"
TEMP_IMG_PATH = r"temp/temp.jpg"
CLASSIFIER_PATH = r"classifiers/haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(CLASSIFIER_PATH)
DETECTION_THRESHOLD = 0.3
DETECTION_TIME_REQUIRED = 1

# Streamlit Config
st.set_page_config(
    page_title="Facial Recognition Attendance System", page_icon="🕗", layout="wide"
)
st.title("Facial Recognition Attendance System")

# Database Connection
conn = st.connection("mysql", type="sql")

# Columns for layout
col1, col2 = st.columns(2)

# Webcam Toggle
if "enable_webcam" not in st.session_state:
    st.session_state.enable_webcam = False

st.session_state.enable_webcam = col1.checkbox(
    "Enable Webcam", value=st.session_state.enable_webcam
)
camera_window = col1.image([])


# Face detection logic
def detect_face(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        img_gray, scaleFactor=1.3, minNeighbors=5, minSize=(200, 200)
    )
    print(f"Number of Faces: {len(faces)}")
    return len(faces) == 1  # Return True if one face is detected, False otherwise


# Process the face detection and recognition
def face_detection(cap):
    detection_start_time = None  # Track the start time when face is detected

    try:
        while st.session_state.enable_webcam:
            ret, img = cap.read()
            if not ret:
                print(f"[WARNING] Failed to capture image.")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            camera_window.image(img_rgb)

            # Check if face is detected
            face_detected = detect_face(img)

            if face_detected:
                if detection_start_time is None:
                    detection_start_time = time.time()  # Start the timer
                elapsed_time = time.time() - detection_start_time

                print(f"[LOG] Elapsed Time: {elapsed_time}")

                # Check if the face has been detected for the required time
                if elapsed_time >= DETECTION_TIME_REQUIRED:
                    cv2.imwrite(TEMP_IMG_PATH, img)
                    try:
                        df = DeepFace.find(
                            img_path=TEMP_IMG_PATH,
                            db_path=DB_PATH,
                            enforce_detection=False,
                            threshold=DETECTION_THRESHOLD,
                            anti_spoofing=True,
                        )
                        if df[0].shape[0] > 0:
                            best_match = df[0]
                            best_match_file_path = best_match["identity"].values[0]
                            best_match_id = os.path.basename(
                                os.path.dirname(best_match_file_path)
                            )

                            # Database Query
                            query = (
                                f"SELECT * FROM profiles WHERE id = {best_match_id};"
                            )
                            db_result = conn.query(query, ttl=60)

                            if db_result.shape[0] == 1:
                                placeholder = col2.empty()
                                placeholder.header(
                                    f"{db_result['last_name'].values[0]}, {db_result['first_name'].values[0]} {db_result['middle_name'].values[0]}",
                                    divider="gray",
                                )
                                time.sleep(5)
                                placeholder.empty()
                            else:
                                col2.write("No matching profile found.")

                        # Reset the detection timer after successful recognition
                        detection_start_time = None

                    except Exception as e:
                        st.error(f"Error in facial recognition: {e}")
                        continue
            else:
                # Reset the detection timer if no face is detected
                detection_start_time = None
    finally:
        cap.release()


# Run webcam if enabled
if st.session_state.enable_webcam:
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    face_detection(cap)
else:
    st.warning("Webcam is disabled.")
