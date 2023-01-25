import streamlit as st

from pathlib import Path

from itertools import combinations
from streamlit_code.csv_processor import process_csv
from streamlit_code.image_processor import process_image
from streamlit_code.pi_group_builder import build_pi_group
from streamlit_code.csv_processor_new import process_csv_new


@st.cache
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


# st.set_page_config(layout="wide")
st.title("Data Processor")

option = st.sidebar.selectbox('Select the type of data to be processed', ('Select an Option', 'Images', 'CSV File', 'CSV File (NEW)', 'Pi Group Builder'))
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

elif option == 'Pi Group Builder':
    build_pi_group()

else:
    st.subheader('Use the side bar to select the type of data you would like to process.')
