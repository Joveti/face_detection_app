import cv2
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Viola-Jones Face Detector", layout="wide")

st.title("Face Detection using Viola-Jones Algorithm")
st.write("This app detects faces in real-time via webcam or from an uploaded image.")

# --- HAAR CASCADE CLASSIFIER ---
# This line ensures the cascade file is found regardless of where the script is run.
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    st.error("Error loading cascade classifier. Make sure the haarcascade file is accessible.")
    st.stop()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Detection Parameters")

# Feature to adjust the scaleFactor parameter
scale_factor = st.sidebar.slider(
    "Scale Factor", 1.01, 1.5, 1.3, 0.01,
    help="How much the image size is reduced at each image scale. Lower is more accurate but slower."
)

# Feature to adjust the minNeighbors parameter
min_neighbors = st.sidebar.slider(
    "Minimum Neighbors", 1, 10, 5, 1,
    help="How many neighbors each candidate rectangle needs to be retained. Higher values reduce false positives."
)

# Feature to choose the color of the rectangles
color_hex = st.sidebar.color_picker("Rectangle Color", "#00FF00")
# Convert hex to BGR tuple for OpenCV (which uses BGR, not RGB)
color_bgr = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0)) # BGR format

# --- APPLICATION MODE SELECTION ---
app_mode = st.sidebar.selectbox("Choose the App Mode",
    ["About", "Upload Image", "Use Webcam"]
)

# --- ABOUT PAGE ---
if app_mode == "About":
    st.markdown("### About This Application")
    st.markdown("""
        This application is an interactive tool for face detection using the **Viola-Jones algorithm**. 
        You can either upload an image or use your webcam for real-time detection.

        **Key Features:**
        - **Two Modes**: Choose between uploading a static image or using a live webcam feed.
        - **Parameter Tuning**: Adjust `scaleFactor` and `minNeighbors` to see how they affect detection accuracy and performance.
        - **Customizable Bounding Box**: Pick any color for the rectangles drawn around detected faces.
        - **Save Results**: Download the processed image with the detected faces.

        To get started, select a mode from the sidebar.
    """)

# --- IMAGE UPLOAD MODE ---
elif app_mode == "Upload Image":
    st.header("Detect Faces from an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=scale_factor, minNeighbors=min_neighbors)

        output_image = image.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(output_image, (x, y), (x + w, y + h), color_bgr, 2)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Original Image', use_column_width=True)
        with col2:
            st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), caption='Image with Detected Faces', use_column_width=True)

        if len(faces) > 0:
            st.success(f"Found {len(faces)} face(s)!")
            _, buf = cv2.imencode(".png", output_image)
            st.download_button(
                label="Download Image",
                data=buf.tobytes(),
                file_name="image_with_faces.png",
                mime="image/png"
            )
        else:
            st.warning("No faces detected. Try adjusting the parameters in the sidebar.")

# --- WEBCAM MODE ---
elif app_mode == "Use Webcam":
    st.header("Detect Faces from Webcam Feed")
    st.write("Click 'START' to begin face detection from your webcam.")

    # This class processes each frame from the webcam
    class FaceDetector(VideoTransformerBase):
        def __init__(self):
            self.scale_factor = 1.3
            self.min_neighbors = 5
            self.color_bgr = (0, 255, 0)

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=self.scale_factor, 
                minNeighbors=self.min_neighbors
            )
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), self.color_bgr, 2)
            
            return img

    # The webrtc_streamer component handles the webcam feed
    ctx = webrtc_streamer(
        key="face-detection",
        video_transformer_factory=FaceDetector,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    )

    # Update the transformer properties in real-time from the sidebar controls
    if ctx.video_transformer:
        ctx.video_transformer.scale_factor = scale_factor
        ctx.video_transformer.min_neighbors = min_neighbors
        ctx.video_transformer.color_bgr = color_bgr