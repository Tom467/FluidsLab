
import copy
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pathlib import Path
from itertools import product
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.axes_grid1 import make_axes_locatable
from general_dimensional_analysis.data_reader import Data
from general_dimensional_analysis.parameter import Parameter
from general_dimensional_analysis.fluid_types import fluid_types, common_constants
from general_dimensional_analysis.group_of_parameter import GroupOfParameters


@st.cache_data
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()


class Plotter:
    def __init__(self, labels=None, cutoff=0.7):
        self.available_markers = ['o', 'v', 's', 'p', '8', '*', 'h', 'x', 'd', '1', '^', 'P', '2', '3', '<', '>', 'H']
        self.available_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#9acd32'] * 2
        self.labels_to_markers = None
        self.labels_to_colors = None
        self.show_legend = True
        self.legend_location = None
        self.legend_font = 10
        self.xlabel_font = 10
        self.ylabel_font = 10
        self.xtick_font = 8
        self.ytick_font = 8
        self.labels = None
        self.masks = None
        if self.labels:
            self.set_labels(labels)
        self.cutoff = cutoff
        self.size = None
        self.color_label = None
        self.color = None
        self.show_regression = False

    def set_labels(self, labels):
        self.labels = labels
        self.labels_to_markers = {label: self.available_markers[i] for i, label in enumerate(set(self.labels))}
        self.set_colors()

    def set_colors(self):
        self.labels_to_colors = {label: self.available_colors[i] for i, label in enumerate(self.labels_to_markers)}

    def set_masks(self, labels):
        self.masks = {a: [True if b == a else False for b in labels] for a in set(labels)}

    def plot(self, x_parameter: Parameter, y_parameter: Parameter):
        # x_parameter, y_parameter, log_x, log_y, color_log, aspect_ratio = Plotter.plot_options(x_parameter, y_parameter)
        fig = plt.figure()
        ax = fig.add_subplot()

        x = x_parameter.values
        y = y_parameter.values

        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and x.shape == y.shape:
            x_pred, y_pred, r_sq = Plotter.best_fit(x, y)
            if r_sq <= self.cutoff:
                pass
            elif st.checkbox(f'Coefficient of Determination: {round(r_sq, 2)}',
                             value=r_sq >= self.cutoff,
                             key=y_parameter.name+'vs'+x_parameter.name):
                log_x, log_y, color_log, aspect_ratio, show_regression = self.plot_options(x_parameter, y_parameter)
                if self.labels:
                    for label in self.masks:
                        if len(self.masks[label]) > 0:
                            sc = plt.scatter(x[self.masks[label]],
                                             y[self.masks[label]],
                                             # s=self.size[self.masks[label]] if self.size is not None else None,
                                             color=None if self.color is not None else self.labels_to_colors[label],
                                             c=self.color[self.masks[label]] if self.color is not None else None,
                                             norm=mcolors.LogNorm(vmin=np.min(self.color), vmax=np.max(self.color)) if color_log else mcolors.Normalize(vmin=np.min(self.color), vmax=np.max(self.color),),
                                             marker=self.labels_to_markers[label],
                                             label=label)
                else:
                    sc = plt.scatter(x, y, s=self.size, color=None if self.color is not None else self.available_colors[0], c=self.color, norm=mcolors.LogNorm(vmin=np.min(self.color), vmax=np.max(self.color)) if color_log else mcolors.Normalize(vmin=np.min(self.color), vmax=np.max(self.color)))
                if (self.show_regression and show_regression) or show_regression:
                    plt.plot(x_pred, y_pred, color='purple', label='Regression Model')
                plt.xlabel(x_parameter.get_markdown(), fontsize=self.xlabel_font)
                plt.ylabel(y_parameter.get_markdown(), fontsize=self.ylabel_font)
                plt.xticks(fontsize=self.xtick_font)
                plt.yticks(fontsize=self.ytick_font)
                if self.color is not None:
                    cbar = plt.colorbar(sc)
                    cbar.set_label(self.color_label, rotation=90)
                if log_x:
                    plt.xscale('log')
                if log_y:
                    plt.yscale('log')
                if aspect_ratio:
                    ax.set_aspect('equal', adjustable='box')
                if self.show_legend:
                    plt.legend(fontsize=str(self.legend_font), loc=self.legend_location)
                st.pyplot(plt)

    @staticmethod
    def best_fit(x, y, degree=2):
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


    def plot_options(self, x, y):
        with st.expander('Individual Plot Options'):
            # if st.checkbox('Invert Y', key='invert_y'+y.name+x.name):
            #     y = y ** -1
            # if st.checkbox('Invert X', key='invert_x'+y.name+x.name):
            #     x = x ** -1
            x_log = st.checkbox('Log plot x', key='x'+y.name+x.name)
            y_log = st.checkbox('Log plot y', key='y'+y.name+x.name)
            color_log = st.checkbox('Log Color Scale', key='color'+y.name+x.name)
            aspect_ratio = st.checkbox('Equal Aspect Ration', key='aspect_ratio'+y.name+x.name)
            show_regression = st.checkbox('Show Regression Line', key='regression_line'+y.name+x.name, value=self.show_regression)
        return x_log, y_log, color_log, aspect_ratio, show_regression

    def options(self, group: GroupOfParameters) -> None:
        self.cutoff = st.slider('Regression Cutoff', 0, 100, 70)/100
        col1, col2 = st.columns(2)
        with col1:
            self.show_regression = st.checkbox('Show Regression line', value=False)
            self.show_legend = st.checkbox('Show Legend', value=True)
            self.legend_location = st.selectbox('Legend Location', ['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'])
            self.legend_font = st.number_input('Legend Font Size', min_value=1, value=10, step=1)
            self.xlabel_font = st.number_input('X Label Font Size', min_value=1, value=10, step=1)
            self.ylabel_font = st.number_input('Y Label Font Size', min_value=1, value=10, step=1)
            self.xtick_font = st.number_input('X Tick Font Size', min_value=1, value=8, step=1)
            self.ytick_font = st.number_input('Y Tick Font Size', min_value=1, value=8, step=1)
        #     if st.checkbox('Map Size'):
        #         self.size = group[st.selectbox('Size Map', group)].values
        with col2:
            if st.checkbox('Map Color'):
                self.color_label = st.selectbox('Color Map', group)
                self.color = group[self.color_label].values

        with st.expander('Colors'):
            self.available_colors = st.text_input('CSS Colors', help='Enter CSS color codes seperated by commas', value='#1f77b4, #ff7f0e, #2ca02c, #d62728, #9467bd, #8c564b, #e377c2, #7f7f7f, #bcbd22, #17becf, #9acd32, #7fff00').replace(' ','').split(',')
            for color in self.available_colors:
                st.markdown(f"<span style='color:{color}'>{color}</span>", unsafe_allow_html=True)
            if len(self.available_colors) < 20:
                self.available_colors = self.available_colors * (20 // len(self.available_colors) + 1)
            self.set_colors()


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
