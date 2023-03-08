
import copy
import itertools
import numpy as np
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from general_dimensional_analysis.data_reader import Data
from general_dimensional_analysis.parameter import Parameter
from general_dimensional_analysis.group_of_parameter import GroupOfParameters


def buckingham_pi_reduction(group, plotter):
    for combo in itertools.combinations(group, group.dimensional_matrix.rank()):
        temp_group = GroupOfParameters(group[parameter] for parameter in combo)
        if temp_group.dimensional_matrix.rank() == group.dimensional_matrix.rank():
            temp_name(temp_group, group-temp_group)


def temp_name(base, remainder):
    pi_groups = []
    for parameter in remainder:
        arr = np.array([0]*remainder.dimensional_matrix.rank()+[1])
        arr[-1] = 1
        pi_groups += [find_nearest_pi_group(base + remainder[parameter], arr)]
        # st.write('Pi Group', parameter, pi_groups[-1])
    group = GroupOfParameters(pi_groups)

    cols = st.columns(len(base.parameters))
    # for parameter in base:
    #     st.markdown(base[parameter].get_markdown())
    for i, parameter in enumerate(base):
        with cols[i]:
            st.subheader(base[parameter].get_markdown())

    titles = [remainder[parameter].get_markdown() for parameter in remainder]
    custom_paiplot(group, titles)


def custom_paiplot(group, titles):
    length = len(group.parameters)
    fig, axs = plt.subplots(length, length)
    for i, parameter1 in enumerate(group):
        for j, parameter2 in enumerate(group):
            if i == j:
                axs[i, j].hist(group[parameter1].values, histtype='step')
            else:
                if group[parameter1].values.shape == group[parameter2].values.shape:
                    axs[i, j].scatter(group[parameter1].values, group[parameter2].values, marker='.', sizes=[2]*len(group[parameter2].values))
            axs[i, j].tick_params(labelsize=5)
            t = axs[i, j].yaxis.get_offset_text()
            t.set_size(5)
            t = axs[i, j].xaxis.get_offset_text()
            t.set_size(5)
            # st.write(group[parameter1].get_markdown())
            axs[i, j].set_ylabel(group[parameter1].get_markdown(), fontsize=8)
            axs[i, j].set_xlabel(group[parameter2].get_markdown(), fontsize=8)
            if i == 0:
                axs[i, j].set_title(titles[j])

    for ax in axs.flat:
        ax.label_outer()

    st.pyplot(plt)

def pairplot(group):
    df = Data.group_to_dataframe_without_units(group)
    options = list(df.columns)
    options.insert(0, None)
    selected = st.sidebar.selectbox('Variable in data to map plot aspects to different colors', options)
    st.pyplot(sns.pairplot(df, hue=selected))


def find_nearest_pi_group(group, arr):
    matrix = group.dimensional_matrix
    # st.write(np.array(matrix, dtype=float))
    # st.write(matrix[:, -1])
    arr = list(-np.round(np.array(matrix[:, :-1].LUsolve(matrix[:, -1]), dtype=np.float64).flatten(), 5)) + [1]
    print(arr)
    arr = find_int(arr)
    print(arr)
    return Parameter.create_from_formula({group[param]: int(arr[i]) for i, param in enumerate(group)})
    # nodes = [arr]
    # counter = 1
    # limit = 4000
    # pi_group = None
    # while nodes and counter < limit:
    #     node = nodes.pop(0)
    #
    #     for i, param in enumerate(node):
    #         node_copy = copy.deepcopy(node)
    #         node_copy[i] += 1
    #         nodes.append(node_copy)
    #
    #         node_copy2 = copy.deepcopy(node)
    #         node_copy2[i] -= 1
    #         nodes.append(node_copy2)
    #
    #     if (matrix @ node == np.zeros(matrix.shape[0])).all() and node.astype(bool).any():
    #         pi_group = Parameter.create_from_formula({group[param]: int(node[i]) for i, param in enumerate(group)})
    #         break
    #     counter += 1
    #     # error
    # if counter == limit:
    #     st.write('was not able to find a pi group with in give limit')
    # return pi_group


def find_int(arr):
    if (np.array(arr) % 1 == np.zeros_like(np.array(arr))).all():
        return arr
    test = False
    epsilon = 1e-15
    maxint = 1000
    for i in range(2, maxint, 1):
        for item in arr:
            if abs(i*item-round(i*item)) < epsilon:
                test = True
            else:
                test = False
                break
        if test:
            return [int(round(i*item)) for item in arr]
    st.write("Could not find one a Pi Group with integer exponents")
    return arr
