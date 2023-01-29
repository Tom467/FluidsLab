
import pandas as pd
import streamlit as st

from pathlib import Path
from itertools import combinations
from streamlit_code.sandbox import sandbox_chart
from streamlit_code.pair_plot import pairplot
from streamlit_code.nullspace import explore_nullspace
from streamlit_code.csv_processor import process_csv
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


# st.set_page_config(layout="wide")
st.title("Data Processor")

option = st.sidebar.selectbox('Select the type of data to be processed', ('Select an Option',
                                                                          'Images',
                                                                          'CSV File',
                                                                          'CSV File (NEW)',
                                                                          'Nullspace',
                                                                          'Sandbox',
                                                                          'Combine Pi Groups',
                                                                          'Pair Plot'))
file = None
if option == 'CSV File':
    instructions = 'Upload a CSV file. Make sure the first row contains header information which should have the following formate: Name-units (e.g. Gravity-acceleration). Also avoid values of zero in the data set as this tends to lead to division by zero.'
    st.subheader('Dimensional Analysis')
    with st.expander('What is Dimensional Analysis?'):
        intro_markdown = read_markdown_file("readme.md")
        st.markdown(intro_markdown)
    with st.expander('Instructions'):
        st.markdown(instructions)
    process_csv(instructions)

elif option == 'CSV File (NEW)':
    instructions = 'Upload a CSV file. Make sure the first row contains header information which should have the following formate: Name-units (e.g. Gravity-acceleration). Also avoid values of zero in the data set as this tends to lead to division by zero.'
    st.subheader('Dimensional Analysis')
    with st.expander('What is Dimensional Analysis?'):
        intro_markdown = read_markdown_file("readme.md")
        st.markdown(intro_markdown)
    with st.expander('Instructions'):
        st.markdown(instructions)
    process_csv_new(instructions)

elif option == 'Images':
    st.subheader('Edges Detection and Contour Tracking')
    with st.expander('Instructions'):
        st.write('Upload images and then use the sliders to select the image and threshold values.')
    process_image()

elif option == 'Nullspace':
    explore_nullspace()

elif option == 'Sandbox':
    sandbox_chart()

elif option == 'Combine Pi Groups':
    group = csv_uploader()
    if group:
        combine_pi_groups(group)

elif option == 'Pair Plot':
    pairplot()

else:
    st.subheader('Use the side bar to select the type of data you would like to process.')

