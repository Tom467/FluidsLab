import copy
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from itertools import product, combinations_with_replacement
from general_dimensional_analysis.util import Util
from general_dimensional_analysis.unit import Unit
from general_dimensional_analysis.parameter import Parameter
from general_dimensional_analysis.group_of_parameter import GroupOfParameters


def step(arr, directions, length):
    while arr and len(arr[0]) < length:
        temp1 = arr.pop(0)
        for i in directions:
            temp2 = copy.deepcopy(temp1)
            temp2.append(i)
            arr.append(temp2)
            if len(temp2) == length:
                yield temp2


def best_pi_groups(group: GroupOfParameters, pi_group_formulas: list, include: list, exclude=[]) -> list:
    show_pi_groups = []
    for pi_group_name in pi_group_formulas:
        check = True
        for param in include:
            if pi_group_name[param.name] == 0:
                check = False
                break

        if check:
            check2 = True
            for param in exclude:
                if pi_group_name[param.name] != 0:
                    check2 = False
            if check2:
                show_pi_groups.append(pi_group_name)

    min_total = 100
    best_param = []
    for formula in show_pi_groups:
        total = 0
        sudo_total = 0
        for i in formula:
            total += abs(formula[i])
            sudo_total += formula[i]
        if total < min_total:
            if len(best_param) > 5:
                min_total = total
                best_param.pop(0)
            new_formula = {}
            for param_name in formula:
                new_formula |= {group[param_name]: -formula[param_name]}
            best_param.append(Parameter.create_from_formula(new_formula))
    for j in copy.deepcopy(best_param):
        print('j', j, j is None)
        if j is None:
            best_param.remove(j)
    print('best', best_param)
    return best_param


def explore_paths(group: GroupOfParameters) -> list:
    nullspace = np.array(group.dimensional_matrix.nullspace()).squeeze()
    pi_group_formulas = []
    for combo in step([[-1], [0], [1]], [-1, 0, 1], nullspace.shape[1]):
        linear_combo = nullspace.T @ nullspace @ np.array(combo)
        formula = {}
        for i, param_name in enumerate(group):
            formula |= {group[param_name]: int(linear_combo[i])}
        pi_group_formulas.append(formula)
    return pi_group_formulas

    # combos = combinations_with_replacement(group, limit)
    #
    # pi_groups = []
    # for param_list in combos:
    #     new_group = GroupOfParameters([group[param_name] for param_name in param_list])
    #     options = []
    #     first = True
    #     for param in new_group:
    #         exp = param_list.count(param)
    #         if first:
    #             options.append([{group[param]: exp}])
    #             first = False
    #         else:
    #             options.append([{group[param]: exp}, {group[param]: 0}, {group[param]: -exp}])
    #
    #     combos = product(*options)
    #     for combo in combos:
    #         a = combo
    #         b = {}
    #         for i in a:
    #             b |= i
    #         new_parameter = Parameter.create_from_formula(b)
    #         if new_parameter.units == Unit(1) and new_parameter not in pi_groups:
    #             pi_groups.append(new_parameter)
    # return pi_groups


if __name__ == '__main__':
    m, l, t = Unit(5), Unit(3), Unit(2)

    a = np.array([np.random.rand()] * 19)
    M = Parameter('M', m, a)
    A = Parameter('Area', l ** 2, a)
    Chord = Parameter('Chord', l, a)
    Span = Parameter('Span', l, a)
    Descent = Parameter('Descent', l / t, a)
    Rotational = Parameter('Rotational', t ** -1, a)
    Angle = Parameter('Angle', Unit(1), a)
    R = Parameter('R', l, a)
    g = Parameter('g', l / t ** 2, a)
    Density = Parameter('Density', m / l ** 3, a)

    parameter_group = GroupOfParameters([Chord, Span, Descent, Rotational, M, A, R, Angle, g, Density])
    include_to_parameters = [Span, Density]
    limit_stop = 4

    generated_pi_groups = explore_paths(parameter_group)

