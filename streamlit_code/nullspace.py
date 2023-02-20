
import copy
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from itertools import combinations, product
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from general_dimensional_analysis.data_reader import Data
from general_dimensional_analysis.parameter import Parameter
from general_dimensional_analysis.group_of_parameter import GroupOfParameters
from general_dimensional_analysis.dimensional_analysis import explore_paths, best_pi_groups


def explore_nullspace(parameter_group, plotter):
    y_include, x_include, shared = define_workspace(parameter_group)

    # st.write(parameter_group)
    pi_group_formulas = get_formulas(parameter_group)

    y_pi_groups = best_pi_groups(parameter_group, pi_group_formulas, y_include+shared, exclude=x_include)
    x_pi_groups = best_pi_groups(parameter_group, pi_group_formulas, x_include+shared, exclude=y_include)
    x, y = pi_group_selector(y_pi_groups, x_pi_groups)

    for combo in product(x, y):
        plotter.plot(combo[0], combo[1], key=combo[0].name+combo[1].name)


def plot(x_parameter: Parameter, y_parameter: Parameter, cutoff=0, key=''):
    with st.expander('Plotting Options'):
        if st.checkbox('Invert Y', key='y'+y_parameter.name+key):
            y_parameter = y_parameter ** -1
        if st.checkbox('Invert X', key='x'+x_parameter.name+key):
            x_parameter = x_parameter ** -1
    legend = []
    plt.figure()

    x = x_parameter.values
    x_pred = np.linspace(np.min(x), np.max(x), 20)
    x_label = x_parameter.get_markdown()

    y = y_parameter.values
    y_label = y_parameter.get_markdown()

    if not isinstance(x, float) and not isinstance(y, float):
        model = LinearRegression().fit(x.reshape((-1, 1)), y)
        r_sq = model.score(x.reshape((-1, 1)), y)

        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly_features = poly.fit_transform(x.reshape(-1, 1))
        poly_model = LinearRegression()
        poly_model.fit(poly_features, y)
        poly_r_sq = poly_model.score(poly_features, y)

        if r_sq >= 0.9 * poly_r_sq:
            y_pred = model.predict(x_pred.reshape((-1, 1)))
        else:
            r_sq = poly_r_sq
            y_pred = poly_model.predict(poly.fit_transform(x_pred.reshape(-1, 1)))

        if r_sq < 1 and not np.all(np.round(y, 2) == np.round(1/x, 2)) and st.checkbox(f'Coefficient of Determination: {round(r_sq, 2)}', value=r_sq >= cutoff, key=key):
            plt.scatter(x, y)
            plt.plot(x_pred, y_pred, color='purple')
            legend.append('Linear Fit')
            plt.xlabel(x_label, fontsize=14)
            plt.ylabel(y_label, fontsize=14)
            st.pyplot(plt)


def define_workspace(dataset: GroupOfParameters):
    share = st.sidebar.checkbox('Share across axes')
    if share:
        col1, col2, col3 = st.sidebar.columns(3)
        with col3:
            st.write('Shared')
    else:
        col1, col2 = st.sidebar.columns(2)
    with col1:
        st.write('Y Axis')
    with col2:
        st.write('X Axis')

    y_params = []
    x_params = []
    shared_param = []
    for i, param in enumerate(dataset.parameters):
        with col1:
            y_box = st.checkbox(param, value=False)
            if y_box:
                y_params.append(dataset.parameters[param])

        with col2:
            if st.checkbox(param, value=False, key=param + 'x', disabled=y_box) and not y_box:
                x_params.append(dataset.parameters[param])

        if share:
            with col3:
                if st.checkbox(param, value=False, key=param + 'shared', disabled=y_box) and not y_box:
                    shared_param.append(dataset.parameters[param])

    return y_params, x_params, shared_param


def get_formulas(group):
    nullspace = np.array(group.dimensional_matrix.nullspace()).squeeze()
    pi_group_formulas = []
    for combo in step([[-1], [0], [1]], [-1, 0, 1], nullspace.shape[1]):
        linear_combo = nullspace.T @ nullspace @ np.array(combo)
        formula = {}
        for i, param_name in enumerate(group):
            formula |= {param_name: int(linear_combo[i])}
        pi_group_formulas.append(formula)
    return pi_group_formulas


def step(arr, directions, length):
    while arr and len(arr[0]) < length:
        temp1 = arr.pop(0)
        for i in directions:
            temp2 = copy.deepcopy(temp1)
            temp2.append(i)
            arr.append(temp2)
            if len(temp2) == length:
                yield temp2


def pi_group_selector(y_pi_groups, x_pi_groups):
    st.sidebar.header('Select Pi groups for X and Y axes')
    col1, col2 = st.sidebar.columns(2)
    counter = 0
    x, y = [], []
    with col1:
        for group in reversed(y_pi_groups):
            counter += 1
            if st.checkbox(group.get_markdown(), key=group.name + 'y' + str(counter)):
                y.append(group)
            st.markdown(group.get_markdown())
    with col2:
        for group in reversed(x_pi_groups):
            counter += 1
            if st.checkbox(group.get_markdown(), key=group.name + 'x' + str(counter)):
                x.append(group)
            st.markdown(group.get_markdown())
    return x, y
