
import copy
import numpy as np
import streamlit as st

from general_dimensional_analysis.parameter import Parameter


def sandbox_chart(group, plotter):
        st.sidebar.header('X Axis')
        x_key = 'x_axis'
        cols = st.sidebar.columns(len(group.parameters))
        if x_key not in st.session_state or len(st.session_state[x_key]) != len(group.parameters):
            st.session_state[x_key] = np.zeros(len(group.parameters), dtype=int)
        for i, param in enumerate(group):
            with cols[i]:
                if st.button('+1', key=param + 'Next' + x_key):
                    st.session_state[x_key][i] += 1
                if st.button('-1', key=param + 'Previous' + x_key):
                    st.session_state[x_key][i] -= 1
                name = param if param[0] == "\\" or (len(param) < 2) else param[0]
                st.write(f'${name}:$')
                st.write(f'${st.session_state[x_key][i]}$')
        dimensioned_x = Parameter.create_from_formula({group[param]: int(st.session_state[x_key][i]) for i, param in enumerate(group)})

        st.sidebar.header('Y Axis')
        key = 'y_axis'
        cols = st.sidebar.columns(len(group.parameters))
        if key not in st.session_state or len(st.session_state[key]) != len(group.parameters):
            st.session_state[key] = np.zeros(len(group.parameters), dtype=int)
        for i, param in enumerate(group):
            with cols[i]:
                if st.button('+1', key=param + 'Next' + key):
                    st.session_state[key][i] += 1
                if st.button('-1', key=param + 'Previous' + key):
                    st.session_state[key][i] -= 1
                name = param if param[0] == "\\" or (len(param) < 2) else param[0]
                st.write(f'${name}:$')
                st.write(f'${st.session_state[key][i]}$')
        dimensioned_y = Parameter.create_from_formula({group[param]: int(st.session_state[key][i]) for i, param in enumerate(group)})

        old_cutoff = plotter.cutoff
        plotter.cutoff = 0
        plotter.plot(dimensioned_x, dimensioned_y)
        if plotter.plot_saving:
            st.subheader('Saved Plots:')
            for saved in reversed(plotter.saved_plots):
                if bool(plotter.saved_plots.get(saved)) and (dimensioned_x != plotter.saved_plots[saved][0] or dimensioned_y != plotter.saved_plots[saved][1]):
                    plotter.plot(*plotter.saved_plots[saved])
        plotter.cutoff = old_cutoff

        # length = 4
        # x_limit = st.sidebar.number_input('X search limit', min_value=1000, value=2000, step=500)
        # x_nearest = find_nearest_pi_group(group, st.session_state[x_key], return_item_length=length, limit=x_limit)
        # # st.write(st.)
        # cols = st.sidebar.columns(2)
        # for i, param in enumerate(x_nearest):
        #     with cols[i % 2]:
        #         if st.checkbox(x_nearest[i].get_markdown()):
        #             x_axis.append(x_nearest[i])
        #         st.write(x_nearest[i].get_markdown())
        #
        # length = 4
        # y_limit = st.sidebar.number_input('Y search limit', min_value=1000, value=2000, step=500)
        # y_nearest = find_nearest_pi_group(group, st.session_state[key], return_item_length=length, limit=y_limit)
        #
        # cols = st.sidebar.columns(2)
        # for i, param in enumerate(y_nearest):
        #     with cols[i % 2]:
        #         if st.checkbox(y_nearest[i].get_markdown(), key=y_nearest[i].get_markdown() + 'y'):
        #             y_axis.append(y_nearest[i])
        #         st.write(y_nearest[i].get_markdown())
        #
        # # cutoff = st.slider('Linear Regression Filter', min_value=0, max_value=100, value=0) / 100
        # for x, y in product(x_axis, y_axis):
        #     if x.name and y.name and x.name != y.name:
        #         plotter.plot(x, y)


@st.cache_data
def find_nearest_pi_group(_group, arr, return_item_length=5, limit=2000):
    matrix = _group.dimensional_matrix
    nodes = [arr]
    nearest_pi_groups = []
    counter = 0
    while nodes and len(nearest_pi_groups) < return_item_length and counter < limit:
        node = nodes.pop(0)

        for i, param in enumerate(node):
            node_copy = copy.deepcopy(node)
            node_copy[i] += 1
            nodes.append(node_copy)

            node_copy2 = copy.deepcopy(node)
            node_copy2[i] -= 1
            nodes.append(node_copy2)
        if (matrix @ node == np.zeros(matrix.shape[0])).all() and node.astype(bool)[arr.astype(bool)].all():
            temp_param = Parameter.create_from_formula({_group[param]: int(node[i]) for i, param in enumerate(_group)})
            if temp_param not in nearest_pi_groups:
                nearest_pi_groups.append(temp_param)
        counter += 1
    return nearest_pi_groups
