
import copy
import numpy as np
import sympy as sp
import streamlit as st

from util import Util
from itertools import product
from itertools import combinations
from parameter import ListOfParameters
from buckingham_pi_theorem.pi_group import PiGroup
from buckingham_pi_theorem.dimensional_matrix import DimensionalMatrix
from general_dimensional_analysis.dimensional_analysis import DimensionalAnalysis


def build_pi_group(analysis: DimensionalAnalysis):
    col1, col2, col3 = st.sidebar.columns(3)
    include, exclude = [], []
    with col1:
        st.write('Include')
        for parameter in analysis:
            if st.checkbox(parameter.name, value=True):
                include.append(parameter)
    with col2:
        st.write('Exclude')
        for parameter in analysis:
            if st.checkbox(parameter.name, key=parameter.name+'exclude'):
                exclude.append(parameter)

    group = ListOfParameters(include)  # analysis.parameters - ListOfParameters(exclude)
    st.write(group)
    M = sp.Matrix(DimensionalMatrix(group).M)
    null = np.array(M.nullspace())
    test = sp.Matrix(null.T[0]).T  # .rref()[0]

    # st.write(null)
    st.write(np.array(test))
    null = np.array(np.array(test)).astype(np.float64)
    total = 0
    for v in null:
        # st.write(str(v))
        total += v
        st.write(pi_group(group, np.round(v, 2)))
    # st.write(pi_group(group, np.round(null[0], 2)))
    # st.write(pi_group(group, np.round(null[1], 2)))
    # st.write(pi_group(group, np.round(null[0]-null[1], 2)))
    # st.write(str(total.T))


def pi_group(parameters, exponents):
    # st.write(str(exponents))
    top = ''
    bottom = ''
    for i, parameter in enumerate(parameters):
        if parameter.name == 'Area':
            print(exponents[i])
        if exponents[i] > 0:
            if exponents[i] == 1:
                top += f'({parameter.name})'
            else:
                top += f'({parameter.name}^'+'{'+f'{int(exponents[i]) if exponents[i] % 1 == 0 else exponents[i]}'+'})'
        elif exponents[i] < 0:
            if exponents[i] == -1:
                bottom += f'({parameter.name})'
            else:
                bottom += f'({parameter.name}^'+'{'+f'{-int(exponents[i]) if exponents[i] % 1 == 0 else -exponents[i]}'+'})'
    if top == '(b_!)':
        print('Error: cannot use b_! as parameter name')
    return r'$\frac{t}{b_!}$'.replace('t', top).replace('b_!', bottom) if bottom else top


def test(analysis, include, exclude):
    variations = [[param, param**-1] for param in include]
    products = list(product(*variations))
    lowest_number = 10000
    for param_combo in products:
        param = param_combo[0]
        for item in param_combo[1:]:
            param *= item
        if param.units.n + param.units.d < lowest_number:
            lowest_number = param.units.n + param.units.d
            best_param = param
    available_parameters = analysis.parameters - ListOfParameters(exclude)
    M = DimensionalMatrix(include)
    # temp = include[0] if include else []
    # if len(include) > M.rank:
    #     for param in include[1:]:
    #         temp *= param

    available_parameters -= ListOfParameters(include)
    available_parameters = best_param + available_parameters
    include = [best_param]

    combos = list(combinations(available_parameters, M.rank+1))
    temp2 = copy.deepcopy(combos)
    for group in combos:
        st.write(ListOfParameters(include), ListOfParameters(group))
        if check(include, group):
            temp2.remove(group)

    pi_groups = []
    st.write('len before', len(temp2))
    for group in copy.deepcopy(temp2):
        add = True
        pi_group = PiGroup(ListOfParameters(group))
        # st.write('start')
        # st.write(pi_group.formula, ListOfParameters(group))
        for i, param in enumerate(include):
            if not pi_group.contains(param.name):
                temp2.remove(pi_group.parameters)
                add = False
                break
        if add:
            pi_groups.append(pi_group)
    st.write('len after', len(pi_groups))
    with col3:
        for group in pi_groups:
            st.checkbox(group.formula)
            st.write(group.formula)


def check(include, group):
    if not Util.list_in_list(include, group):
        return True
    if DimensionalMatrix(group[1:]).rank < DimensionalMatrix(group).rank:
        return True
    st.write(DimensionalMatrix(group[1:]).M)
    if DimensionalMatrix(group[1:]).det is None or DimensionalMatrix(group[1:]).det == 0:
        return True
    return False
