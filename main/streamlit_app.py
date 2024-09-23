import streamlit as st
from PIL import Image
import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLO model
model = YOLO("C:\\Users\\SDC-03\\Desktop\\image_detection\\main\\runs\\detect\\train28\\weights\\best.pt")

# Streamlit app
st.title("Tool Detection")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Perform prediction
    results = model.predict(source=opencv_image)

    # Display the results
    results_image = results[0].plot()

    st.image(results_image, caption='Processed Image', use_column_width=True)
    st.write(results)

# To run
# streamlit run streamlit_app.py --server.address 172.18.101.47 --server.port 8505
