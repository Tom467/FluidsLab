
import copy
import numpy as np

from general_dimensional_analysis.parameter import Parameter
from general_dimensional_analysis.data_reader import Data
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


if __name__ == '__main__':
    experiment = Data("C:/Users/truma/Downloads/FormattedData.csv")
    group = experiment.parameters
    R = group['R']
    M = group['M']
    Span = group['Span']
    Area = group['Area']
    Angle = group['Angle']
    Chord = group['Chord']
    Descent = group['Descent']
    Rotational = group['Rotational']

    print(Angle.values)

