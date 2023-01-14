
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from data_reader import Data
from streamlit_code.plotting import plot
from buckingham_pi_theorem.dimensional_analysis import DimensionalAnalysis


def generate_plots(dimensional_analysis, markers, my_bar, inverting, hide_plot):
    plt.close('all')
    for h, pi_group_set in enumerate(dimensional_analysis.pi_group_sets):
        text = pi_group_set.repeating_variables[0].name
        for repeating in pi_group_set.repeating_variables[1:]:
            text += r', a'.replace('a', repeating.name)
        with st.expander(text, expanded=True):
            for i, pi_group in enumerate(pi_group_set.pi_groups[1:]):
                key = f'Set: {h + 1} Group: {i + 1}'
                if hide_plot:
                    plot(pi_group, pi_group_set, markers, key, inverting)
                else:
                    plot(pi_group, pi_group_set, markers, key, inverting)
        my_bar.progress((h+1) / len(dimensional_analysis.pi_group_sets))


def process_csv(instructions):
    file = st.sidebar.file_uploader('CSV file', type=['csv'], help=instructions)

    if file is not None:
        ds = pd.read_csv(file)
        st.sidebar.write("Dataset Preview:")
        st.sidebar.write(ds)

        markers = []

        complete_dimensional_analysis = True
        option = st.sidebar.selectbox('Options', ['Select...', 'Add Variable', 'Modify Workspace', 'Add Labels'])
        if option != 'Select...':
            complete_dimensional_analysis = False
        if option == 'Add Variable':
            add_parameter(ds)
        elif option == 'Add Labels':
            markers = add_markers(markers)
        if option == 'Modify Workspace':
            modified_workspace, repeating_variables = modify_workspace(ds)
        else:
            modified_workspace, repeating_variables = Data(ds).parameters, Data(ds).parameters
        print(not complete_dimensional_analysis)
        with st.expander('Plotting Options', expanded=not complete_dimensional_analysis):
            complete_dimensional_analysis = st.checkbox('Dimensional Analysis', value=complete_dimensional_analysis)
            inverting = st.checkbox('Allow Inverting', value=False)
            hide_plot = st.checkbox('Hide Individual Plots', value=False)

        if complete_dimensional_analysis:
            d = DimensionalAnalysis(modified_workspace, repeating_parameters=repeating_variables)
            st.subheader('Generating Possible Figures')
            my_bar = st.progress(0)
            st.write('Different Sets of Repeating Variables')
            generate_plots(d, markers, my_bar, inverting, hide_plot)
        # st.balloons()


def add_parameter(ds):
    st.sidebar.text_input('Variable Name', placeholder='Variable Name')
    # st.sidebar.selectbox('Variable Units', Units().get_units())
    selected = None
    if selected is not None:
        operator_selector()
    else:
        selected = parameter_selector(ds)


def parameter_selector(ds):
    col1, col2, col3 = st.sidebar.columns(3)
    for i, param in enumerate(ds):
        if i % 3 == 0:
            with col1:
                if st.button(param.split('-')[0]):
                    return param
        elif i % 3 == 1:
            with col2:
                if st.button(param.split('-')[0]):
                    return param
        elif i % 3 == 2:
            with col3:
                if st.button(param.split('-')[0]):
                    return param
    return None


def operator_selector():
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        st.button('*')



def modify_workspace(dataset):
    dataset_modified = dataset
    print('1st', type(dataset_modified))
    st.sidebar.write('Dependent Variable')
    dependent_variable = st.sidebar.selectbox('Dependent Variable', [param for param in dataset], label_visibility='collapsed')
    arr = []
    for i, param in enumerate(dataset):
        if param == dependent_variable:
            arr = [i] + arr
        else:
            arr.append(i)
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.write('Independent Variables')
    with col2:
        st.write('Repeating Variables')

    counter = 0
    for i, param in enumerate(dataset):
        with col1:
            acceptable = st.checkbox(param, value=True, disabled=(param == dependent_variable)) if param != dependent_variable else True
            # print(param, dependent_variable)
            if not acceptable:
                dataset_modified = dataset_modified.drop(param, axis=1)
                arr.remove(i-counter)
                counter += 1
                arr = [num-1 if num > i-counter else num for num in arr]
    repeating_variables = dataset_modified

    for param in dataset_modified:
        with col2:
            if dependent_variable == param or not st.checkbox(param, value=True, key=param+'repeating'):
                repeating_variables = repeating_variables.drop(param, axis=1)
    print(type(dataset_modified))
    data = Data(dataset_modified.iloc[:, arr])
    repeating_var = Data(repeating_variables)
    return data.parameters, repeating_var.parameters


def add_markers(markers):
    available_markers = {'point': '.', 'circle': 'o', 'triangle': 'v', 'square': 's', 'pentagon': 'p', 'octagon': '8',
                         'star': '*', 'hexagon': 'h', 'x': 'x', 'diamond': 'd'}

    number = st.sidebar.text_input('Number of Markers', value=0, placeholder='0',
                                   help='Select the number of markers to label subgroups in the data.')
    for i in range(int(number)):
        label = st.sidebar.text_input(f'Label {i + 1}', placeholder=f'label for group {i + 1}')
        col1, col2 = st.sidebar.columns(2)
        with col1:
            first = int(st.text_input(f'Group {i + 1} Beginning', value=0))
        with col2:
            last = int(st.text_input(f'Group {i + 1} End', value=0))
        marker_type = st.sidebar.selectbox(f'Marker {i + 1}', available_markers, label_visibility='collapsed')
        markers.append([first, last, available_markers[marker_type], label])

    return markers
