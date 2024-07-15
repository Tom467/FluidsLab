import copy

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from itertools import product
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from general_dimensional_analysis.data_reader import Data
from general_dimensional_analysis.parameter import Parameter
from general_dimensional_analysis.group_of_parameter import GroupOfParameters
from general_dimensional_analysis.dimensional_analysis import explore_paths


def process_csv_new(instructions):
    file = st.sidebar.file_uploader('CSV file', type=['csv'], help=instructions)

    if file is not None:
        ds = pd.read_csv(file)
        with st.sidebar.expander('Dataset Preview:'):
            st.write(ds)
        parameter_group = Data(ds).parameters
        y_include, x_include = define_workspace(parameter_group)
        y_pi_groups, x_pi_groups = generate_pi_groups(parameter_group, y_include, x_include, limit=4)

        col1, col2 = st.sidebar.columns(2)
        x, y = [], []
        with col1:
            for group in y_pi_groups:
                if st.checkbox(group.get_markdown()):
                    y.append(group)
                st.markdown(group.get_markdown())
        with col2:
            for group in x_pi_groups:
                if st.checkbox(group.get_markdown(), key=group.name + 'x'):
                    x.append(group)
                st.markdown(group.get_markdown())
        cutoff = st.slider('Linear Regression Filter', min_value=0, max_value=100, value=0) / 100
        for a, b in product(x, y):
            plot(a, b, str(a)+str(b), cutoff)
        # print('y_groups:', y_pi_groups)
        # print('x_groups:', x_pi_groups)


def define_workspace(dataset: GroupOfParameters):
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.write('Y Axis')
    with col2:
        st.write('X Axis')

    y_params = []
    x_params = []
    for i, param in enumerate(dataset.parameters):
        with col1:
            y_box = st.checkbox(param, value=False)
            if y_box:
                y_params.append(dataset.parameters[param])

        with col2:
            if st.checkbox(param, value=False, key=param + 'x', disabled=y_box) and not y_box:
                x_params.append(dataset.parameters[param])

    return y_params, x_params


def plot(x_parameter: Parameter, y_parameter: Parameter, key: str, cutoff):

    legend = []
    plt.figure()

    x = x_parameter.values
    x_pred = np.linspace(np.min(x), np.max(x), 20)
    x_label = x_parameter.get_markdown()

    y = y_parameter.values
    y_label = y_parameter.get_markdown()

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


# @st.cache
def generate_pi_groups(parameter_group, y_include, x_include, limit=None):
    if limit:
        pass
    else:
        limit = len(y_include) + 2
    print('y include', y_include, 'x_include', x_include)
    x_parameter_group = parameter_group - y_include
    print('x group', x_parameter_group)
    y_parameter_group = parameter_group - x_include
    print('y group', y_parameter_group)
    set1 = get_pi_groups(x_parameter_group, limit)
    set2 = get_pi_groups(y_parameter_group, limit)
    y_pi_groups, x_pi_groups = [], []

    for pi_group in set1:
        check_x = True
        for param in x_include:
            if param not in pi_group.formula:
                check_x = False

        if check_x:
            x_pi_groups.append(pi_group)

    for pi_group in set2:
        check_y = True

        for param in y_include:
            if param not in pi_group.formula:
                check_y = False
        if check_y:
            y_pi_groups.append(pi_group)

    return y_pi_groups, x_pi_groups


@st.cache_data
def get_pi_groups(parameter_group, limit):
    return explore_paths(parameter_group, limit=limit)
