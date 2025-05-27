# Facial Recognition App

A mini project to try out facial recognition, inspired by the HRIS Attendance System from the university I used to work at. 

At first, I used `cv2.VideoCapture` for the camera, but I learned the hard way that [it doesn't work](https://discuss.streamlit.io/t/webcam-not-opening-in-share-streamlit/49180) on Streamlit Community Cloud. So, I had to switch things up and use `streamlit-webrtc` instead.
