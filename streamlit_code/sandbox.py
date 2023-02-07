
import copy
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from itertools import product, combinations
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from streamlit_code.streamlit_util import plot, saved_plots
from general_dimensional_analysis.data_reader import Data
from general_dimensional_analysis.parameter import Parameter
from general_dimensional_analysis.group_of_parameter import GroupOfParameters


def sandbox_chart():
    file = st.sidebar.file_uploader('CSV file', type=['csv'])

    if file is not None:
        ds = pd.read_csv(file)
        group = Data(ds).parameters
        create_pi_groups(group)


def create_pi_groups(group):

        x_axis, y_axis = [], []

        st.sidebar.header('X Axis')
        key = 'x_axis'
        cols = st.sidebar.columns(len(group.parameters))
        if key not in st.session_state:
            st.session_state[key] = np.zeros(len(group.parameters), dtype=int)
        for i, param in enumerate(group):
            with cols[i]:
                if st.button('+1', key=param + 'Next' + key):
                    st.session_state[key][i] += 1
                if st.button('-1', key=param + 'Previous' + key):
                    st.session_state[key][i] -= 1
                st.write(f'${param if len(param) < 4 else param[:4]}:{st.session_state[key][i]}$')
        dimensioned_x = Parameter.create_from_formula({group[param]: int(st.session_state[key][i]) for i, param in enumerate(group)})
        length = 4
        x_nearest = find_nearest_pi_group(group, st.session_state[key], return_item_length=length)
        cols = st.sidebar.columns(len(x_nearest)+1)
        for i, col in enumerate(cols[:-1]):
            with col:
                if st.checkbox(x_nearest[i].get_markdown()):
                    x_axis.append(x_nearest[i])
                st.write(x_nearest[i].get_markdown())

        st.sidebar.header('Y Axis')
        key = 'y_axis'
        cols = st.sidebar.columns(len(group.parameters))
        if key not in st.session_state:
            st.session_state[key] = np.zeros(len(group.parameters), dtype=int)
        for i, param in enumerate(group):
            with cols[i]:
                if st.button('+1', key=param + 'Next' + key):
                    st.session_state[key][i] += 1
                if st.button('-1', key=param + 'Previous' + key):
                    st.session_state[key][i] -= 1
                st.write(f'${param if len(param) < 4 else param[:4]}: {st.session_state[key][i]}$')
        dimensioned_y = Parameter.create_from_formula({group[param]: int(st.session_state[key][i]) for i, param in enumerate(group)})
        length = 4
        y_nearest = find_nearest_pi_group(group, st.session_state[key], return_item_length=length)
        cols = st.sidebar.columns(len(y_nearest)+1)
        for i, col in enumerate(cols[:-1]):
            with col:
                if st.checkbox(y_nearest[i].get_markdown(), key=y_nearest[i].get_markdown()+'y'):
                    y_axis.append(y_nearest[i])
                st.write(y_nearest[i].get_markdown())

        # cutoff = st.slider('Linear Regression Filter', min_value=0, max_value=100, value=0) / 100
        plot(dimensioned_x, dimensioned_y)
        for x, y in product(x_axis, y_axis):
            plot(x, y, key=x.name+y.name)


def formula_generator(group: GroupOfParameters, key='') -> Parameter:
    cols = st.sidebar.columns(len(group.parameters))
    formula = {}
    for i, param in enumerate(group):
        if param+key not in st.session_state:
            st.session_state[param+key] = 0
        with cols[i]:
            if st.button('+1', key=param+'Next'+key):
                st.session_state[param+key] += 1
            st.write(f'{param}: {st.session_state[param+key]}')
            if st.button('-1', key=param+'Previous'+key):
                st.session_state[param+key] -= 1

        formula |= {group[param]: st.session_state[param+key]}
    st.write(Parameter.create_from_formula(formula))
    return Parameter.create_from_formula(formula)


def find_nearest_pi_group(group, arr, return_item_length=5):
    matrix = group.dimensional_matrix
    nodes = [arr]
    nearest_pi_groups = []
    counter = 0
    while nodes and len(nearest_pi_groups) < return_item_length and counter < 2000:
        node = nodes.pop(0)

        for i, param in enumerate(node):
            node_copy = copy.deepcopy(node)
            node_copy[i] += 1
            nodes.append(node_copy)

            node_copy2 = copy.deepcopy(node)
            node_copy2[i] -= 1
            nodes.append(node_copy2)
        if (matrix @ node == np.zeros(matrix.shape[0])).all() and node.astype(bool)[arr.astype(bool)].all():
            temp_param = Parameter.create_from_formula({group[param]: int(node[i]) for i, param in enumerate(group)})
            if temp_param not in nearest_pi_groups:
                nearest_pi_groups.append(temp_param)
        counter += 1
    return nearest_pi_groups


# nodes = step([[0], [1], [-1]], [0, 1, -1], len(parameter_group.parameters))
#
# st.header("X Axis")
# if 'index_x' not in st.session_state:
#     st.session_state.index_x = 0
# if st.button('Next', key='Next x'):
#     st.session_state.index_x += 1
# cols = st.columns(len(parameter_group.parameters))
# if st.button('Previous', key='Prev x'):
#     st.session_state.index_x -= 1
#
# formula = {}
# for i, param in enumerate(parameter_group):
#     with cols[i]:
#         st.write(f'{param}')
#         exp = int(st.text_input('exponent', value=0, key=param + 'x', label_visibility="collapsed"))
#     formula |= {parameter_group[param]: nodes[st.session_state.index_x][i] + exp}
# x = Parameter.create_from_formula(formula)
# # x = formula_generator(parameter_group, 'x', nodes[st.session_state.index_x])
#
# st.header("Y Axis")
# if 'index_y' not in st.session_state:
#     st.session_state['index_y'] = 0
# if st.button('Next', key='Next y'):
#     st.session_state['index_y'] += 1
# cols = st.columns(len(parameter_group.parameters))
# if st.button('Previous', key='Prev y'):
#     st.session_state['index_y'] -= 1
# formula = {}
# for i, param in enumerate(parameter_group):
#     with cols[i]:
#         st.write(f'{param}')
#         exp = int(st.text_input('exponent', value=0, key=param + 'y', label_visibility="collapsed"))
#     formula |= {parameter_group[param]: nodes[st.session_state.index_y][i] + exp}
# y = Parameter.create_from_formula(formula)
# # y = formula_generator(parameter_group, 'y', nodes[st.session_state.index_y])