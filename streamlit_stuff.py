
import streamlit as st

from pathlib import Path
from streamlit_code.sandbox import sandbox_chart
from streamlit_code.pair_plot import pairplot
from streamlit_code.nullspace import explore_nullspace
from streamlit_code.csv_processor import process_csv
from streamlit_code.streamlit_util import add_constants, Plotter
from streamlit_code.image_processor import process_image
from streamlit_code.auto_exploration import explore_pi_groups
from general_dimensional_analysis.data_reader import Data
from governing_equations.navier_stokes import NavierStokes


st.set_page_config(page_title="Data Processor", layout="wide")


@st.cache_data
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


@st.cache_data
def file_reader(file):
    dataframe = Data.csv_to_dataframe(file)
    group, label = Data.dataframe_to_group(dataframe)
    return group, label, dataframe


def image_options(files):
    st.subheader('Edges Detection and Contour Tracking')
    with st.expander('Instructions'):
        st.write('Upload images and then use the sliders to select the image and threshold values.')
    process_image(files)


def csv_options(file):
    st.subheader('Dimensional Analysis')
    # with st.expander('What is Dimensional Analysis?'):
    #     intro_markdown = read_markdown_file("readme.md")
    #     st.markdown(intro_markdown)

    tab1, tab2, tab3 = st.tabs(['Analysis', 'Plot Options', 'Data Table'])

    p = Plotter()

    with tab3:
        group, label, dataframe = file_reader(file)
        st.dataframe(dataframe)

    with tab2:
        p.options(group)
        p.set_labels(label)
    supplemental_group = add_constants(group)
    option = st.sidebar.selectbox('Select the type of data to be processed', ('Select an Option',
                                                                              'Pair Plot',
                                                                              'Sandbox',
                                                                              'Auto Exploration'))
    with tab1:
        if option == 'Buckingham Pi':
            process_csv()

        elif option == 'Sandbox':
            sandbox_chart(supplemental_group, p)

        elif option == 'Auto Exploration':
            explore_pi_groups(supplemental_group, p)

        elif option == 'Pair Plot':
            pairplot(group)


def governing_equations():
    st.subheader('Find the Governing Equations')
    ns = NavierStokes()
    assumptions = st.text_input('Assumptions', value='2D, no gravity, steady')
    assumptions = [assume.strip() for assume in assumptions.split(',')]
    eqn_x, eqn_y, eqn_z = ns.simplify_naiver(assumptions)
    st.markdown('The $Navier$-$Stokes$ equations are:')
    st.markdown(f'x-direction: {eqn_x}')
    st.markdown(f'y-direction: {eqn_y}')
    st.markdown(f'z-direction: {eqn_z}')


# st.set_page_config(layout="wide")
st.title("Data Processor")

uploaded_file = st.sidebar.file_uploader('CSV file', type=['csv', 'tif', 'png', 'jpg'], accept_multiple_files=True)

if uploaded_file:
    if uploaded_file[0].name[-3:] == 'csv':
        csv_options(uploaded_file[0])
    else:
        image_options(uploaded_file)
else:
    instructions = 'Upload a CSV file. Make sure the first row contains header information which should have the following formate: Name-units (e.g. Gravity-acceleration). Also avoid values of zero in the data set as this tends to lead to division by zero.'
    with st.expander('Instructions'):
        st.markdown(instructions)

