
import json
import copy
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pathlib import Path
from itertools import product
from bokeh.plotting import figure
from scipy.optimize import curve_fit
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


def power_fit(x, a):
    return x ** a


class Plotter:
    def __init__(self, labels=None, cutoff=0.7):
        self.available_markers = ['circle', 'square', 'triangle', 'star', 'plus', 'hex', 'diamond', 'cross', 'circle_dot', 'diamond_dot', 'square_dot', 'hex_dot', 'star_dot', 'triangle_dot', 'square_pin', 'triangle_pin'] * 2
        self.available_markers2 = ['o', 'v', 's', 'p', '8', '*', 'h', 'x', 'd', '1', '^', 'P', '2', '3', '<', '>', 'H']
        self.available_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#9acd32'] * 4
        self.labels_to_markers = None
        self.labels_to_colors = None
        self.marker_transparency = 0.6
        self.font = None
        self.show_legend = True
        self.legend_location = None
        self.legend_size = 10
        self.xlabel_size = 10
        self.ylabel_size = 10
        self.marker_size = 10
        self.xtick_size = 8
        self.ytick_size = 8
        self.labels = None
        self.masks = None
        if self.labels:
            self.set_labels(labels)
        self.cutoff = cutoff
        self.size = None
        self.color_label = None
        self.color = None
        self.show_regression = False
        self.show_power_law = True
        self.plot_saving = True
        if 'plots' not in st.session_state:
            st.session_state['plots'] = {}
        self.saved_plots = st.session_state['plots']

    def set_labels(self, labels):
        self.labels = labels
        self.labels_to_markers = {label: self.available_markers[i] for i, label in enumerate(set(self.labels))}
        self.set_colors()

    def set_colors(self):
        self.labels_to_colors = {label: self.available_colors[i] for i, label in enumerate(self.labels_to_markers)}

    def set_masks(self, labels):
        self.masks = {a: [True if b == a else False for b in labels] for a in set(labels)}

    def plot(self, x_parameter: Parameter, y_parameter: Parameter):
        x = x_parameter.values
        y = y_parameter.values
        # ax = fig.add_subplot()

        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and x.shape == y.shape:

            x_pred, y_pred, r_sq = Plotter.best_fit(x, y)
            if r_sq <= self.cutoff:
                pass
            elif st.checkbox(f'Coefficient of Determination: {round(r_sq, 2)}',
                             value=r_sq >= self.cutoff,
                             key=y_parameter.name+'vs'+x_parameter.name):

                log_x, log_y, color_log, aspect_ratio, show_regression, show_power = self.plot_options(x_parameter, y_parameter)

                p = figure(
                    x_axis_label='$' + x_parameter.get_markdown() + '$',
                    y_axis_label='$' + y_parameter.get_markdown() + '$',
                    x_axis_type="log" if log_x else "linear",
                    y_axis_type="log" if log_y else "linear", )

                if self.labels:
                    for label in self.masks:
                        if len(self.masks[label]) > 0:
                            scatter = getattr(p, self.labels_to_markers[label])
                            scatter(
                                x[self.masks[label]],
                                y[self.masks[label]],
                                # s=self.size[self.masks[label]] if self.size is not None else None,
                                # fill_color=None if self.color is not None else self.labels_to_colors[label],
                                color=self.labels_to_colors[label],
                                size=self.marker_size,
                                alpha=self.marker_transparency,
                                # norm=mcolors.LogNorm(vmin=np.min(self.color), vmax=np.max(self.color)) if color_log else mcolors.Normalize(vmin=np.min(self.color), vmax=np.max(self.color),),
                                # marker=self.labels_to_markers[label],
                                legend_label=label if self.show_legend else None,
                            )
                else:
                    p.circle(
                        x,
                        y,
                        # s=self.size,
                        color=self.available_colors[0],
                        size=self.marker_size,
                        alpha=self.marker_transparency,
                        # color=None if self.color is not None else self.available_colors[0],
                        # c=self.color,
                        # norm=mcolors.LogNorm(vmin=np.min(self.color), vmax=np.max(self.color)) if color_log else mcolors.Normalize(vmin=np.min(self.color), vmax=np.max(self.color))
                        x_axis_type="log" if log_x else "linear",
                        y_axis_type="log" if log_y else "linear",
                    )
                if (self.show_regression and show_regression) or show_regression:
                    p.line(x_pred, y_pred, line_color='purple', legend_label='Regression Model' if self.show_legend else None)
                if (self.show_power_law and show_power) or show_power:
                    power = curve_fit(power_fit, x, y, p0=[1])
                    if np.isinf(power[1]):
                        power = curve_fit(power_fit, x, y, p0=[-1])
                    p.line(x_pred, x_pred**power[0], legend_label=f'Power Law {power[0][0]:.2f}' if self.show_legend else None)
                # plt.xlabel(x_parameter.get_markdown(), fontsize=self.xlabel_size)
                # plt.ylabel(y_parameter.get_markdown(), fontsize=self.ylabel_size)
                # plt.xticks(fontsize=self.xtick_size)
                # plt.yticks(fontsize=self.ytick_size)
                # if self.color is not None:
                #     cbar = plt.colorbar(sc)
                #     cbar.set_label(self.color_label, rotation=90)
                # if log_x:
                #     plt.xscale('log')
                # if log_y:
                #     plt.yscale('log')
                # if aspect_ratio:
                #     ax.set_aspect('equal', adjustable='box')
                # if self.show_legend:
                #     p.legend(fontsize=str(self.legend_size), loc=self.legend_location)

                p.xaxis.axis_label_text_font_size = str(self.xlabel_size)+'pt'
                p.yaxis.axis_label_text_font_size = str(self.ylabel_size)+'pt'
                p.xaxis.major_label_text_font_size = str(self.xtick_size)+'pt'
                p.yaxis.major_label_text_font_size = str(self.ytick_size)+'pt'
                p.legend.label_text_font = "times"
                p.legend.label_text_font_size = str(self.legend_size)+'pt'
                p.legend.location = self.legend_location
                p.grid.grid_line_color = None

                p.xaxis.major_label_orientation = np.pi / 4
                # p.yaxis.major_label_orientation = np.pi / 4
                st.bokeh_chart(p, use_container_width=True)

    def plot2(self, x_parameter: Parameter, y_parameter: Parameter):
        x = x_parameter.values
        y = y_parameter.values
        fig = plt.figure()
        ax = fig.add_subplot()

        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and x.shape == y.shape:

            x_pred, y_pred, r_sq = Plotter.best_fit(x, y)
            if r_sq <= self.cutoff:
                pass
            elif st.checkbox(f'Coefficient of Determination: {round(r_sq, 2)}',
                             value=r_sq >= self.cutoff,
                             key=y_parameter.name+'vs'+x_parameter.name):

                log_x, log_y, color_log, aspect_ratio, show_regression, show_power = self.plot_options(x_parameter, y_parameter)

                if self.labels:
                    for label in self.masks:
                        if len(self.masks[label]) > 0:
                            sc = plt.scatter(
                                x[self.masks[label]],
                                y[self.masks[label]],
                                # s=self.size[self.masks[label]] if self.size is not None else None,
                                color=None if self.color is not None else self.labels_to_colors[label],
                                c=self.color[self.masks[label]] if self.color is not None else None,
                                norm=mcolors.LogNorm(vmin=np.min(self.color), vmax=np.max(self.color)) if color_log else mcolors.Normalize(vmin=np.min(self.color), vmax=np.max(self.color),),
                                marker=self.labels_to_markers[label],
                                label=label
                            )
                else:
                    sc = plt.scatter(
                        x,
                        y,
                        s=self.size,
                        color=None if self.color is not None else self.available_colors[0],
                        c=self.color,
                        norm=mcolors.LogNorm(vmin=np.min(self.color), vmax=np.max(self.color)) if color_log else mcolors.Normalize(vmin=np.min(self.color), vmax=np.max(self.color))
                    )
                if (self.show_regression and show_regression) or show_regression:
                    plt.plot(x_pred, y_pred, color='purple', label='Regression Model')
                if (self.show_power_law and show_power) or show_power:
                    power = curve_fit(power_fit, x, y, p0=[1])
                    if np.isinf(power[1]):
                        power = curve_fit(power_fit, x, y, p0=[-1])
                    plt.plot(x_pred, x_pred**power[0], label=f'Power Law {power[0][0]:.2f}')
                plt.xlabel(x_parameter.get_markdown(), fontsize=self.xlabel_size)
                plt.ylabel(y_parameter.get_markdown(), fontsize=self.ylabel_size)
                plt.xticks(fontsize=self.xtick_size)
                plt.yticks(fontsize=self.ytick_size)
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
                    plt.legend(fontsize=str(self.legend_size), loc=self.legend_location)
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
            if self.plot_saving:
                saved = st.checkbox('Save Plot', key=x.name + 'vs' + y.name + 'save', value=self.saved_plots.get(f'{x.name}, {y.name}'), help='After unsaving a plot, it will not immediately disappear.')
                if saved:
                    self.saved_plots[f'{x.name}, {y.name}'] = x, y
                else:
                    # st.write('need to get rid of: ')
                    # st.write(self.saved_plots[f'{x.name}, {y.name}'])
                    self.saved_plots[f'{x.name}, {y.name}'] = None

            # if st.checkbox('Invert Y', key='invert_y'+y.name+x.name):
            #     y = y ** -1
            # if st.checkbox('Invert X', key='invert_x'+y.name+x.name):
            #     x = x ** -1
            x_log = st.checkbox('Log plot x', key='x'+y.name+x.name)
            y_log = st.checkbox('Log plot y', key='y'+y.name+x.name)
            color_log = False  # st.checkbox('Log Color Scale', key='color'+y.name+x.name)
            aspect_ratio = False  # st.checkbox('Equal Aspect Ratio', key='aspect_ratio'+y.name+x.name)
            show_regression = st.checkbox('Show Regression Line', key='regression_line'+y.name+x.name, value=self.show_regression)
            show_power = st.checkbox('Show Power Law', key='power_law'+y.name+x.name, value=self.show_power_law)
        return x_log, y_log, color_log, aspect_ratio, show_regression, show_power

    def options(self, group: GroupOfParameters) -> None:
        self.cutoff = st.slider('Regression Cutoff', 0, 100, 70)/100
        col1, col2 = st.columns(2)
        with col1:
            self.show_regression = st.checkbox('Show Regression line', value=False)
            self.show_power_law = st.checkbox('Show Power Law', value=True)
            self.show_legend = st.checkbox('Show Legend', value=True)
            self.legend_location = st.selectbox('Legend Location', ['top_left', 'top_center', 'top_right', 'center_right', 'bottom_right', 'bottom_center', 'bottom_left', 'center_left', 'center'])
            self.legend_size = st.number_input('Legend Font Size', min_value=1, value=10, step=1)
            self.xlabel_size = st.number_input('X Label Font Size', min_value=1, value=10, step=1)
            self.ylabel_size = st.number_input('Y Label Font Size', min_value=1, value=10, step=1)
            self.xtick_size = st.number_input('X Tick Font Size', min_value=1, value=8, step=1)
            self.ytick_size = st.number_input('Y Tick Font Size', min_value=1, value=8, step=1)
            self.marker_size = st.number_input('Marker Size', min_value=1, value=8, step=1)
            # self.font = st.selectbox('Font', matplotlib.font_manager.get_font_names())

        #     if st.checkbox('Map Size'):
        #         self.size = group[st.selectbox('Size Map', group)].values
        with col2:
            self.plot_saving = st.checkbox('Allow Plot Saving (Does not work properly with filtering data by label)', value=True)
            if st.checkbox('Map with Color'):
                self.color_label = st.selectbox('Color Map', group)
                self.color = group[self.color_label].values

            self.marker_transparency = st.number_input('Marker Transparency', min_value=0.0, max_value=1.0, value=0.6, step=0.1)
            with st.expander('Colors'):
                self.available_colors = st.text_input('CSS Colors', help='Enter CSS color codes seperated by commas', value='#1f77b4, #ff7f0e, #2ca02c, #d62728, #9467bd, #8c564b, #e377c2, #7f7f7f, #bcbd22, #17becf, #9acd32, #7fff00').replace(' ','').split(',')
                samples = ''
                for color in self.available_colors:
                    samples += f"<span style='color:{color}'>{color}</span>, "
                st.markdown(samples, unsafe_allow_html=True)
                if len(self.available_colors) < 20:
                    self.available_colors = self.available_colors * (20 // len(self.available_colors) + 1)
                self.set_colors()

        settings = {
            'available_markers': self.available_markers,
            'available_color': self.available_colors,
            'marker_transparency': self.marker_transparency,
            'font': self.font,
            'show_legend': self.show_legend,
            'legend_location': self.legend_location,
            'legend_size': self.legend_size,
            'xlabel_size': self.xlabel_size,
            'ylabel_size': self.ylabel_size,
            'marker_size': self.marker_size,
            'xtick_size': self.xtick_size,
            'ytick_size': self.ytick_size,
            'cutoff': self.cutoff,
            'show_regression': self.show_regression,
            'show_power_law': self.show_power_law,
            'plot_saving': self.plot_saving,
        }
        # st.download_button('Download Plot Settings', json.dumps(settings), 'plot_options.json')

    def add_to_saved_plots(self, x_parameter, y_parameter):
        pass

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
