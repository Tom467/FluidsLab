import numpy as np
import streamlit as st
from numpy import sin, arcsin, cos, arccos, tan, arctan, pi, sqrt, log, exp, deg2rad
sind = lambda degrees: sin(deg2rad(degrees))
cosd = lambda degrees: cos(deg2rad(degrees))
from pathlib import Path
from streamlit_code.sandbox import sandbox_chart
from streamlit_code.pair_plot import pairplot
from streamlit_code.nullspace import explore_nullspace
from streamlit_code.csv_processor import process_csv
from streamlit_code.buckingham_pi import buckingham_pi_reduction
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
    st.subheader('Original Data')
    st.dataframe(dataframe)
    return df_to_group(dataframe), dataframe


@st.cache_data
def df_to_group(dataframe):
    group, labels = Data.dataframe_to_group(dataframe)
    return group, labels


def image_options(files):
    st.subheader('Edges Detection and Contour Tracking')
    with st.expander('Instructions'):
        st.write('Upload images and then select the number of operations to perform on the image. A typical place to start is to crop the image to show just the region of interest, then search for edges with canny.')
    process_image(files)


def csv_options(file):
    st.subheader('Dimensional Analysis')
    # with st.expander('What is Dimensional Analysis?'):
    #     intro_markdown = read_markdown_file("readme.md")
    #     st.markdown(intro_markdown)

    tab1, tab2, tab3 = st.tabs(['Analysis', 'Plot Options', 'Data Table'])

    p = Plotter()
    with tab3:
        (group, labels), dataframe = file_reader(file)
        if 'operation_dict' not in st.session_state:
            st.session_state['operation_dict'] = {name[0]: ('x', lambda x: x) for name in dataframe.columns}
        include_labels = []
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Modify')
            selected = st.selectbox('Edit', [name[0] for name in dataframe.columns])
            # st.write(st.session_state['operation_dict'][selected][0])
            operation = st.text_input('formula', value=st.session_state['operation_dict'][selected][0], key=selected, help='Use x as the variable to represent the variable being modified. Availbe functions include: sin, sind, arcsin, cos, cosd, arcos, tan, arctan, pi, sqrt, log, and exp. Note: to raise a value to a power use ** instead of the commonly used ^.')
            if operation == '':
                operation = 'x'
            st.session_state['operation_dict'][selected] = operation, lambda x: eval(operation)
        # with col2:
        #     st.subheader('Add')
        #     name = st.text_input('Name')
        #     units = st.text_input('Units')
        #     input_values = st.text_input('Values')
        #     if input_values:
        #         values = np.array(input_values.split(','), dtype=float)
        #         dataframe[name, units] = values
        with col2:
            if labels:
                st.write('True')
                st.subheader('Filter')
                for parameter in st.session_state['operation_dict']:
                    dataframe[parameter] = dataframe[parameter].apply(st.session_state['operation_dict'][parameter][1])
                for label in set(labels):
                    if st.checkbox(f'{label}', value=True):
                        include_labels.append(label)
                mask = [item in include_labels for item in dataframe['Label'].values]

        st.subheader('Edited Data')
        dataframe = st.experimental_data_editor(dataframe[mask]) if labels else st.experimental_data_editor(dataframe)
        group, labels = df_to_group(dataframe)

    with tab2:
        p.options(group)
        p.set_labels(labels)
    supplemented_group = add_constants(group)
    option = st.sidebar.selectbox('Select the type of analysis to be completed', ('Select an Option',
                                                                                  'Pair Plot',
                                                                                  'Sandbox',
                                                                                  'Buckingham Pi',
                                                                                  'Auto Exploration'))
    with tab1:
        # if option == 'Buckingham Pi':
        #     process_csv()

        if option == 'Sandbox':
            sandbox_chart(supplemented_group, p)

        elif option == 'Buckingham Pi':
            buckingham_pi_reduction(supplemented_group, p)

        elif option == 'Auto Exploration':
            explore_pi_groups(supplemented_group, p)

        elif option == 'Pair Plot':
            pairplot(group)


def governing_equations():
    st.subheader('Find the Governing Equations')
    ns = NavierStokes()
    assumptions = st.text_input('Assumptions', value='2D, no gravity, steady', help="Acceptable assumptions: no gravity, steady, u=0, 2D, constant pressure, inviscid, external flow")
    assumptions = [assume.strip() for assume in assumptions.split(',')]
    eqn_x, eqn_y, eqn_z = ns.simplify_naiver(assumptions)
    st.markdown('The $Navier$-$Stokes$ equations are:')
    st.markdown(f'x-direction: {eqn_x}')
    st.markdown(f'y-direction: {eqn_y}')
    st.markdown(f'z-direction: {eqn_z}')


# st.set_page_config(layout="wide")
st.title("Data Processor")

uploaded_file = st.sidebar.file_uploader('File', type=['csv', 'tif', 'png', 'jpg', 'jpeg'], accept_multiple_files=True)

if uploaded_file:
    if uploaded_file[0].name[-3:] == 'csv':
        csv_options(uploaded_file[0])
    else:
        image_options(uploaded_file)
else:
    instructions = 'Upload either images or data in a csv file'
    with st.expander('Instructions'):
        st.markdown(instructions)
    with st.expander('BETA feature: Governing Equations'):
        governing_equations()
