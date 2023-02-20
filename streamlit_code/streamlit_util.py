
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


class Plotter:
    def __init__(self, labels=None, cutoff=0.7):
        self.available_markers = ['o', 'v', 's', 'p', '8', '*', 'h', 'x', 'd', '1', '^', 'P', '2', '3', '<', '>', 'H']
        self.labels_to_markers = None
        self.labels = None
        self.masks = None
        if self.labels:
            self.set_labels(labels)
        self.cutoff = cutoff
        self.size = None
        self.color = None

    def set_labels(self, labels):
        self.labels = labels
        self.labels_to_markers = {label: self.available_markers[i] for i, label in enumerate(set(self.labels))}
        self.masks = {a: [True if b == a else False for b in self.labels] for a in self.labels_to_markers}

    def plot(self, x_parameter: Parameter, y_parameter: Parameter):
        # x_parameter, y_parameter = Plotter.plot_options(x_parameter, y_parameter)
        plt.figure()

        x = x_parameter.values
        y = y_parameter.values

        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            x_pred, y_pred, r_sq = self.best_fit(x, y)
            if r_sq <= self.cutoff:
                pass
            elif st.checkbox(f'Coefficient of Determination: {round(r_sq, 2)}', value=r_sq >= self.cutoff, key=y_parameter.name+'vs'+x_parameter.name):
                if self.labels:
                    for label in self.labels_to_markers:
                        plt.scatter(x[self.masks[label]],
                                    y[self.masks[label]],
                                    s=self.size[self.masks[label]] if self.size is not None else None,
                                    c=self.color[self.masks[label]] if self.color is not None else None,
                                    marker=self.labels_to_markers[label],
                                    label=label)
                else:
                    plt.scatter(x, y, s=self.size, c=self.color)
                plt.plot(x_pred, y_pred, color='purple', label='Regression Model')
                plt.xlabel(x_parameter.get_markdown(), fontsize=14)
                plt.ylabel(y_parameter.get_markdown(), fontsize=14)
                plt.legend()
                st.pyplot(plt)

    @st.cache_data
    def best_fit(self, x, y, degree=2):
        model = LinearRegression().fit(x.reshape((-1, 1)), y)
        r_sq = model.score(x.reshape((-1, 1)), y)

        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(x.reshape(-1, 1))
        poly_model = LinearRegression()
        poly_model.fit(poly_features, y)
        poly_r_sq = poly_model.score(poly_features, y)

        x_pred = np.linspace(np.min(x), np.max(x), 20)
        if r_sq >= 0.9 * poly_r_sq:
            y_pred = model.predict(x_pred.reshape((-1, 1)))
        else:
            r_sq = poly_r_sq
            y_pred = poly_model.predict(poly.fit_transform(x_pred.reshape(-1, 1)))
        return x_pred, y_pred, r_sq

    @staticmethod
    def plot_options(x, y):
        with st.expander('Plotting Options'):
            if st.checkbox('Invert Y', key='invert_y'+y.name+x.name):
                y = y ** -1
            if st.checkbox('Invert X', key='invert_x'+y.name+x.name):
                x = x ** -1
        return x, y

    def options(self, group: GroupOfParameters) -> None:
        self.cutoff = st.slider('Regression Cutoff', 0, 100, 1)/100
        col1, col2 = st.columns(2)
        with col1:
            if st.checkbox('Map Size'):
                size_map = group[st.selectbox('Size Map', group)].values
                self.size = (size_map - np.min(size_map)) / np.max(size_map) * 100 + 10
        with col2:
            if st.checkbox('Map Color'):
                color_map = group[st.selectbox('Color Map', group)].values
                self.color = (color_map - np.min(color_map)) / np.max(color_map)


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
    selected_constants = GroupOfParameters([])
    with st.sidebar.expander('Add to data'):
        options = list(fluid_types)
        options.insert(0, '')
        selected = st.selectbox('Add fluid type', options)
        if selected:
            other_parameters = fluid_types[selected].parameters

        constants = common_constants + other_parameters
        for constant in constants:
            constant_name = constant.replace('\\', '')
            if st.checkbox(f"{constant_name}: {constants[constant].values[0]}", key=constant+'constant'):
                selected_constants += GroupOfParameters([constants[constant]])
    return group + selected_constants
