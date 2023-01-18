
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from util import Util
from data_reader import Data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from general_dimensional_analysis.dimensional_analysis import DimensionalAnalysis


def process_csv_new(instructions):
    file = st.sidebar.file_uploader('CSV file', type=['csv'], help=instructions)

    if file is not None:
        ds = pd.read_csv(file)
        with st.sidebar.expander('Dataset Preview:'):
            st.write(ds)
        analysis = DimensionalAnalysis(Data(ds).parameters)
        y_params, x_params = define_workspace(analysis)

        x_list, y_list = [], []
        for param in y_params:
            group = analysis.create_pi_groups(param)
            [y_list.append(item) for item in group]
        for param in x_params:
            group = analysis.create_pi_groups(param)
            [x_list.append(item) for item in group]

        if st.checkbox('Show Pi Groups'):
            col1, col2 = st.columns(2)
            with col1:
                st.write('Y Axis')
                for group in y_list:
                    st.write(group.formula)
            with col2:
                st.write('X Axis')
                for group in x_list:
                    st.write(group.formula)

        cutoff = st.slider('Linear Regression Filter', min_value=0, max_value=100, value=70) / 100
        for i, y_param in enumerate(y_list):
            for j, x_param in enumerate(x_list):
                plot(x_param, y_param, str((i, j)), cutoff)



def define_workspace(dataset):
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.write('Y Axis')
    with col2:
        st.write('X Axis')

    y_params = []
    x_params = []
    for i, param in enumerate(dataset.parameters):
        with col1:
            if st.checkbox(param.name, value=False):
                y_params.append(param)

    for param in dataset.parameters:
        with col2:
            if st.checkbox(param.name, value=False, key=param.name+'x'):
                x_params.append(param)

    return y_params, x_params


def plot(x_parameter, y_parameter, key, cutoff=0):
    legend = []
    plt.figure()

    x = x_parameter.values
    x_pred = np.linspace(np.min(x), np.max(x), 20)
    x_label = x_parameter.formula

    y = y_parameter.values
    y_label = y_parameter.formula

    # c = np.stack([x, y], axis=1)
    # c = c[c[:, 0].argsort()]
    # x, y = c[:, 0], c[:, 1]

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

    if r_sq < 1 and not np.all(np.round(y, 2) == np.round(1/x, 2)) and st.checkbox(key + f' - Coefficient of Determination: {round(r_sq, 2)}', value=r_sq >= cutoff):
        plt.scatter(x, y)
        plt.plot(x_pred, y_pred, color='purple')
        legend.append('Linear Fit')
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        st.pyplot(plt)


def show_pi_groups(x_groups, y_groups):
    pass

