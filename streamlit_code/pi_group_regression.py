
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


def combine_pi_groups(group):
    st.write(group)
    st.write(group['Angle'].values)
    pi_group_list = generate_pi_groups(group.dimensional_matrix)
    for x, y in combinations(pi_group_list, 2):
        score = sum(abs(x * y))
        if score == 0:
            x = Parameter.create_from_formula({group[param]: int(x[i]) for i, param in enumerate(group)})
            y = Parameter.create_from_formula({group[param]: int(y[i]) for i, param in enumerate(group)})
            print(x, y)
            r_sq = plot(x, y)
            if r_sq:
                print(r_sq)


@st.cache
def generate_pi_groups(matrix, arr=(0, 1, -1)):
    pi_group_list = []
    for combo in product(np.array(arr), repeat=matrix.shape[1]):
        node = np.array(combo)
        if (matrix @ node == np.zeros(matrix.shape[0])).all() and not (node == np.zeros(matrix.shape[1])).all():
            pi_group_list.append(node)
    return pi_group_list
