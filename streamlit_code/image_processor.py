import cv2
import numpy as np
import streamlit as st

from PIL import Image


def find_contours(img, threshold1=100, threshold2=200, blur=3):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (blur, blur), sigmaX=0, sigmaY=0)
    edges = cv2.Canny(image=img_blur, threshold1=threshold1, threshold2=threshold2)
    return cv2.findContours(image=edges, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE), edges


def process_image(image_files):
    # image_files = st.sidebar.file_uploader('Image Uploader', type=['tif', 'png', 'jpg'], help='Upload .tif files to to test threshold values for Canny edge detection. Note multiple images can be uploaded but there is a 1 GB RAM limit and the application can begin to slow down if more than a couple hundred images are uploaded', accept_multiple_files=True)
    # if len(image_files) > 0:
    image_number = 1
    if len(image_files) > 1:
        image_number = st.sidebar.slider('Image Number', min_value=1, max_value=len(image_files))
    image = np.array(Image.open(image_files[image_number-1]))
    try:
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    except cv2.error:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image_copy = image.copy()

    threshold1 = st.sidebar.slider('Minimum Threshold', min_value=0, max_value=200, value=100, help='Any pixel below this threshold is eliminated, and any above are consider possible edges')
    threshold2 = st.sidebar.slider('Definite Threshold', min_value=0, max_value=200, value=200, help='Any pixel above this threshold is consider a definite edge. Additionally any pixel above the minimum threshold and connected to a pixel already determined to be an edge will be declared an edge')
    blur = st.sidebar.slider('blur', min_value=1, max_value=10, value=2, help='Filters out noise. Note: blur values must be odd so blur_value = 2 x slider_value + 1')

    (contours, _), edge_img = find_contours(image, threshold1=threshold1, threshold2=threshold2, blur=2*blur-1)
    image_copy = cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255, 100, 55), thickness=1, lineType=cv2.LINE_AA)
    if st.sidebar.checkbox("Show just edges"):
        st.image(edge_img)
    else:
        st.image(image_copy)
