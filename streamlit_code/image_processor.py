import cv2
import numpy as np
import streamlit as st

from PIL import Image


def process_image(image_files):
    image_number = 1
    video = False
    if image_files[image_number-1].name[-3:] == 'mp4':
        video = True
        image_files = process_video(image_files[image_number-1])

    col1, col2 = st.columns(2)
    with col2:
        if len(image_files) > 1:
            image_number = st.slider('Image Number', min_value=1, max_value=len(image_files))

    image = image_files[image_number-1] if video else np.array(Image.open(image_files[image_number-1]))
    if image.shape[-1] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    with col1:
        operations = define_operations(image.shape)
    results = perform_operations(image, operations)
    new_image = results['image']
    with col2:
        st.markdown(
            """
            <style>
                [data-testid=stSidebar] [data-testid=stImage]{
                    text-align: center;
                    display: block;
                    margin-left: auto;
                    margin-right: auto;
                    width: 100%;
                }
            </style>
            """, unsafe_allow_html=True
        )
        st.image(new_image)


def write_video(output_file, images, framerate=20, color=False, fourcc=cv2.VideoWriter_fourcc(*'mp4v')):
    video = cv2.VideoWriter(output_file, fourcc, framerate, (images.shape[2], images.shape[1]), color)
    for i in images:
        video.write(i)
    video.release()


@st.cache_data
def process_video(uploaded_video):
    vid = uploaded_video.name
    with open(vid, mode='wb') as f:
        f.write(uploaded_video.read())  # save video to disk

    video = cv2.VideoCapture(vid)
    success = True
    frames = []
    while success:
        success, frame = video.read()
        frames.append(frame)
    # st.video(uploaded_video)
    return frames


def define_operations(image_shape):
    number_of_steps = st.number_input('Number of Steps', step=1, value=1)
    operations = []
    for i in range(number_of_steps):
        with st.expander(f"Step {i+1}", expanded=True):
            # skip = st.checkbox('Skip Step', key='skip'+f'step_{i+1}')
            operations.append(operation_selector(f'Step {i+1}', image_shape))
    return operations


def perform_operations(image, operations):
    image_copy = image.copy()
    for operation in operations:
        image_copy = operation(image_copy)
    return image_copy


def operation_selector(key, image_shape):
    operation = st.selectbox(f'Operation for {key}', ['Crop', 'Canny Edge Detection', 'Threshold'], label_visibility='hidden')
    if operation == 'Crop':
        crop = Crop()
        crop.configure(image_shape, key)
        return crop.process()
    elif operation == 'Canny Edge Detection':
        canny = Canny()
        canny.configure(key)
        return canny.process()
    elif operation == 'Threshold':
        thresh = Threshold()
        thresh.configure(key)
        return thresh.process()


class Crop:
    def __init__(self):
        self.top = None
        self.bottom = None
        self.left = None
        self.right = None

    def configure(self, image_shape, key=''):
        self.top = st.number_input('Top Border', min_value=0, step=50, key='Top Border'+key)
        self.left = st.number_input('Left Border', min_value=0, step=50, key='Left Border'+key)
        self.right = st.number_input('Right Border', max_value=0, step=50, key='Right Border'+key)-1
        self.bottom = st.number_input('Bottom Border', max_value=0, step=50, key='Bottom Border'+key)-1

    def process(self):
        return lambda x: x[self.top: self.bottom, self.left: self.right]


class Canny:
    def __init__(self):
        self.threshold1 = None
        self.threshold2 = None
        self.blur = None

    def configure(self, key):
        self.threshold1 = st.slider('Minimum Threshold', min_value=0, max_value=200, value=100,
                                    key='Minimum Threshold'+key,
                                    help='Any pixel below this threshold is eliminated, and any above are consider '
                                         'possible edges')
        self.threshold2 = st.slider('Definite Threshold', min_value=0, max_value=200, value=200,
                                    key='Definite Threshold'+key,
                                    help='Any pixel above this threshold is consider a definite edge. Additionally any '
                                         'pixel above the minimum threshold and connected to a pixel already determined '
                                         'to be an edge will be declared an edge')
        self.blur = 2 * st.slider('blur', min_value=1, max_value=10, value=2, key='blur'+key,
                                    help='Filters out noise. Note: blur values must be odd so actual blur value = 2 x '
                                         'slider value + 1') + 1

    def process(self):
        a = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
        b = lambda x: cv2.GaussianBlur(a(x), (self.blur, self.blur), sigmaX=0, sigmaY=0)
        c = lambda x: cv2.Canny(image=b(x), threshold1=self.threshold1, threshold2=self.threshold2)
        d = lambda x: cv2.findContours(image=c(x), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)[0]
        e = lambda x: cv2.drawContours(image=x, contours=d(x), contourIdx=-1, color=(255, 100, 55), thickness=1, lineType=cv2.LINE_AA)
        return e


class ConvertColor:
    def __init__(self):
        self.type = {'RGB2GRAY': cv2.COLOR_RGB2GRAY,
                     'BGR2GRAY': cv2.COLOR_BGR2GRAY}

def canny(image, key='', skip=False):

    image_copy = image.copy()
    img_gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (blur, blur), sigmaX=0, sigmaY=0)
    edges = cv2.Canny(image=img_blur, threshold1=threshold1, threshold2=threshold2)
    if st.checkbox('Only show edges', key='edges'+key):
        return edges if not skip else image

    contours, _ = cv2.findContours(image=edges, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    contour_length = st.slider('Minimum Contour Length', min_value=0,
                               max_value=max([len(contour) for contour in contours]),
                               value=100,
                               key='contour_length' + key)
    contours = [contour for contour in contours if len(contour) >= contour_length]
    image_copy = cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(255, 100, 55),
                                  thickness=1, lineType=cv2.LINE_AA)
    if contours:
        st.write(f'Number of contours found: {len(contours)}')
    return image_copy if not skip else image


class Threshold:
    def __init__(self):
        self.simple = True
        self.simple_types = {'THRESH_BINARY': cv2.THRESH_BINARY,
                             'THRESH_BINARY_INV': cv2.THRESH_BINARY_INV,
                             'THRESH_TRUNC': cv2.THRESH_TRUNC,
                             'THRESH_TOZERO': cv2.THRESH_TOZERO,
                             'THRESH_TOZERO_INV': cv2.THRESH_TOZERO_INV}
        self.adaptive_types = {'ADAPTIVE_THRESH_MEAN_C': cv2.ADAPTIVE_THRESH_MEAN_C,
                               'ADAPTIVE_THRESH_GAUSSIAN_C': cv2.ADAPTIVE_THRESH_GAUSSIAN_C}
        self.simple_type = None
        self.adaptive_type = None
        self.otsu = 0
        self.threshold_value = None
        self.block_size = None
        self.constant = None

    def simple_threshold(self, image):
        # st.write(self.simple_type+self.otsu)
        return cv2.cvtColor(cv2.threshold(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), self.threshold_value, 255, self.simple_type+self.otsu)[1], cv2.COLOR_GRAY2RGB)

    def adaptive_threshold(self, image):
        return cv2.cvtColor(cv2.adaptiveThreshold(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 255,
                                     self.adaptive_type,
                                     self.simple_type,
                                     self.block_size,
                                     self.constant), cv2.COLOR_GRAY2RGB)

    def process(self):
        st.write(self.simple)
        if self.simple:
            return self.simple_threshold
        return self.adaptive_threshold

    def configure(self, key):
        self.simple_type = self.simple_types[st.selectbox(f'Threshold type for {key}', self.simple_types)]
        if st.checkbox('Adaptive', key='adaptive' + key, help='The algorithm determines the threshold for a pixel based on a small region around it'):
            self.simple = False
            self.block_size = st.number_input('Block Size', min_value=3, value=195, step=10, key='block_size' + key, help='Determines the size of the neighbourhood area. Note: the algorithm requires an odd value')
            self.constant = st.number_input('Constant', value=5, key='constant' + key, help='A constant that is subtracted from the mean or weighted sum of the neighbourhood pixels')
            self.adaptive_type = self.adaptive_types[st.selectbox(f'Threshold Type', self.adaptive_types, key=f'Threshold type for {key}')]
        else:
            self.simple = True
            self.otsu = cv2.THRESH_OTSU if st.checkbox(f'Otsu Threshold', key='Otsu_Threshold' + key, help="Otsu's method determines an optimal global threshold value from the image histogram") else 0
            self.threshold_value = st.slider('Threshold', min_value=0, max_value=255, value=100, key='Threshold' + key)




class ImageEditor:
    def __init__(self):
        pass
