
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
from general_dimensional_analysis.fluid_types import fluid_types, common_constants
from general_dimensional_analysis.group_of_parameter import GroupOfParameters


def plotting_options(x, y, key):
    with st.expander('Plotting Options'):
        if st.checkbox('Invert Y', key='y'+y.name+x.name+key):
            y = y ** -1
        if st.checkbox('Invert X', key='x'+y.name+x.name+key):
            x = x ** -1
    return x, y


def plot(x_parameter: Parameter, y_parameter: Parameter, cutoff=0, key='', collapse=False, highlight=None):
    if collapse:
        pass
    else:
        x_parameter, y_parameter = plotting_options(x_parameter, y_parameter, key)
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
        if collapse and r_sq <= cutoff:
            pass
        elif st.checkbox(f'Coefficient of Determination: {round(r_sq, 2)}', value=r_sq >= cutoff, key=y_parameter.name+'vs'+x_parameter.name+key):
            plt.scatter(x, y)
            if highlight is not None:
                plt.scatter(x[highlight], y[highlight], marker='*')
            plt.plot(x_pred, y_pred, color='purple')
            legend.append('Linear Fit')
            plt.xlabel(x_label, fontsize=14)
            plt.ylabel(y_label, fontsize=14)
            st.pyplot(plt)


def saved_plots():
    if st.session_state.saved_plots:
        for i, pair in enumerate(st.session_state.saved_plots):
            st.write(pair[0])
            # plot(*pair, key=str(i))


def add_to_saved_plots(item):
    temp_list = []
    if st.session_state.saved_plots:
        temp_list = copy.deepcopy(st.session_state.saved_plots)
    temp_list.append(item)
    st.session_state.saved_plots = temp_list


def add_constants(group):
    other_parameters = GroupOfParameters([])
    with st.sidebar.expander('Add to data'):
        options = list(fluid_types)
        options.insert(0, '')
        selected = st.selectbox('Add fluid type', options)
        if selected:
            other_parameters = fluid_types[selected].parameters

        constants = common_constants
        for constant in constants:
            if st.checkbox(constant, key=constant+'constant'):
                other_parameters += GroupOfParameters([constants[constant]])
        # st.write(other_parameters)
        other = other_parameters
    return group + other if other_parameters else group
