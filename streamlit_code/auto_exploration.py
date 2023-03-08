
import copy
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from itertools import product, combinations
from general_dimensional_analysis.parameter import Parameter
from general_dimensional_analysis.group_of_parameter import GroupOfParameters


def explore_pi_groups(full_group, plotter):
    group = parameter_selector(full_group)
    pi_group_list = generate_pi_groups([parameter for parameter in group], group, arr=(0, 1, -1, 2, -2, 3))
    st.sidebar.write(f'Number of Pi Groups found: {len(pi_group_list)}')

    weights = np.ones(pi_group_list[0].shape)
    st.sidebar.subheader('Select variables that can be shared between axes:')
    for i, param in enumerate(group):
        if st.sidebar.checkbox(param):
            weights[i] = 0

    for x, y in combinations(pi_group_list, 2):
        score = sum(abs(x * y) * weights)
        if score <= 0:
            x = Parameter.create_from_formula({group[param]: int(x[i]) for i, param in enumerate(group)})
            y = Parameter.create_from_formula({group[param]: int(y[i]) for i, param in enumerate(group)})
            plotter.plot(x, y)


@st.cache_data
def generate_pi_groups(parameters, _group, arr=(0, 1, -1, 2, -2)):
    matrix = GroupOfParameters([_group[parameter] for parameter in parameters]).dimensional_matrix
    pi_group_list = []
    for combo in product(np.array(arr), repeat=matrix.shape[1]):
        node = np.array(combo)
        if (matrix @ node == np.zeros(matrix.shape[0])).all():  # and not (node == np.zeros(matrix.shape[1])).all():
            pi_group_list.append(node)
    return pi_group_list


def compare_deviation(pi_groups, group):
    min = [.3]*4
    min_param = [None, None, None, None]
    for x in pi_groups:
        param = Parameter.create_from_formula({group[param]: int(x[i]) for i, param in enumerate(group)})
        test = np.std(param.values) / np.mean(param.values)
        if test < min[0]:
            st.write(param.get_markdown())
            min.pop(0)
            min_param.pop(0)
            min.append(test)
            min_param.append(param)
    # st.write(f'Max std: {max(std)}')
    st.write(f'Min std: {np.round(min, 2)}')
    st.write(min_param)


def parameter_selector(full_group: GroupOfParameters) -> GroupOfParameters:
    include_parameters = []
    st.sidebar.subheader('Select up to 6 variables to include in exploration:')
    for parameter in full_group:
        if st.sidebar.checkbox(parameter, key='selector'+parameter) and len(include_parameters) < 7:
            include_parameters.append(full_group[parameter])
    return GroupOfParameters(include_parameters)

