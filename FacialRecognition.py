# Libraries
import os
import time
import queue

import cv2
import streamlit as st

from av import VideoFrame
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from deepface import DeepFace
from st_supabase_connection import SupabaseConnection

# Constants
DB_PATH = "db"
TEMP_IMG_PATH = r"temp/temp.jpg"
CLASSIFIER_PATH = r"classifiers/haarcascade_frontalface_default.xml"

FACE_CASCADE = cv2.CascadeClassifier(CLASSIFIER_PATH)

DETECTION_THRESHOLD = 0.3
DETECTION_TIME_REQUIRED = 1

# Global state variables
face_detected_since = None
img_since = None
has_checked_db = False

# Streamlit Config
st.set_page_config(page_title="Facial Recognition App", page_icon="🕗")
st.title("Facial Recognition App")

# Database Connection
conn = st.connection("supabase", type=SupabaseConnection)

result_queue = queue.Queue()


# Face detection logic
def detect_face(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(
        img_gray, scaleFactor=1.3, minNeighbors=5, minSize=(200, 200)
    )
    print(f"Number of Faces: {len(faces)}")
    return len(faces) == 1  # Return True if one face is detected, False otherwise


def check_face_database(img):
    cv2.imwrite(TEMP_IMG_PATH, img)

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
        best_match_id = os.path.basename(os.path.dirname(best_match_file_path))

        print(f"Best Match: {best_match_id}")

        db_result = conn.table("profiles").select("*").eq("id", best_match_id).execute()

        result_queue.put(db_result)


def video_frame_callback(frame):
    global face_detected_since, img_since, has_checked_db

    img = frame.to_ndarray(format="bgr24")
    face_detected = detect_face(img)
    current_time = time.time()

    if face_detected:
        if face_detected_since is None and img_since is None:
            face_detected_since = current_time
            img_since = img
        elif (
            current_time - face_detected_since >= DETECTION_TIME_REQUIRED
            and not has_checked_db
        ):
            check_face_database(img_since)
            has_checked_db = True
    else:
        face_detected_since = None
        img_since = None
        has_checked_db = False

    return VideoFrame.from_ndarray(img, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="sample",
    mode=WebRtcMode.SENDRECV,
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
    rtc_configuration={  # Add this config
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
)

if webrtc_ctx.state.playing:
    while True:
        result = result_queue.get()
        st.toast(result)
        time.sleep(5)
