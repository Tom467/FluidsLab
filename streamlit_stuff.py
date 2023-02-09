
import pandas as pd
import streamlit as st

from pathlib import Path
from itertools import combinations
from streamlit_code.sandbox import sandbox_chart
from streamlit_code.pair_plot import pairplot
from streamlit_code.nullspace import explore_nullspace
from streamlit_code.csv_processor import process_csv
from streamlit_code.streamlit_util import add_constants
from streamlit_code.image_processor import process_image
from streamlit_code.csv_processor_new import process_csv_new
from streamlit_code.pi_group_regression import combine_pi_groups
from general_dimensional_analysis.data_reader import Data


st.set_page_config(page_title="Data Processor")


def csv_uploader():
    file = st.sidebar.file_uploader('CSV file', type=['csv'])
    group = None
    if file is not None:
        ds = pd.read_csv(file)
        group = Data(ds).parameters
    return group


@st.cache
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

def image_options(files):
    st.subheader('Edges Detection and Contour Tracking')
    with st.expander('Instructions'):
        st.write('Upload images and then use the sliders to select the image and threshold values.')
    process_image(files)


def csv_options(file):
    st.subheader('Dimensional Analysis')
    with st.expander('What is Dimensional Analysis?'):
        intro_markdown = read_markdown_file("readme.md")
        st.markdown(intro_markdown)

    group = Data.csv_to_group(file)
    supplemental_group = add_constants(group)
    option = st.sidebar.selectbox('Select the type of data to be processed', ('Select an Option',
                                                                              'Buckingham Pi',
                                                                              'Nullspace',
                                                                              'Sandbox',
                                                                              'Combine Pi Groups',
                                                                              'Pair Plot'))
    if option == 'Buckingham Pi':
        process_csv()

    elif option == 'Nullspace':
        explore_nullspace(supplemental_group)

    elif option == 'Sandbox':
        sandbox_chart(supplemental_group)

    elif option == 'Combine Pi Groups':
        combine_pi_groups(supplemental_group)

    elif option == 'Pair Plot':
        pairplot(group)


# st.set_page_config(layout="wide")
st.title("Data Processor")

uploaded_file = st.sidebar.file_uploader('CSV file', type=['csv','tif', 'png', 'jpg'], accept_multiple_files=True)

if uploaded_file:
    if uploaded_file[0].name[-3:] == 'csv':
        csv_options(uploaded_file[0])
    else:
        image_options(uploaded_file)
else:
    instructions = 'Upload a CSV file. Make sure the first row contains header information which should have the following formate: Name-units (e.g. Gravity-acceleration). Also avoid values of zero in the data set as this tends to lead to division by zero.'
    with st.expander('Instructions'):
        st.markdown(instructions)
