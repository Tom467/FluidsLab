import numpy as np

from util import Util
from units import Units
from itertools import combinations
from parameter import ListOfParameters, Parameter
from buckingham_pi_theorem.pi_group import PiGroup
from buckingham_pi_theorem.dimensional_matrix import DimensionalMatrix


class DimensionalAnalysis:
    def __init__(self, parameters, arr=None):
        arr = arr if arr is not None else [2, 2, 2] + [0] * (len(parameters)-3)
        self.parameters = parameters
        self.free_parameters, self.input_parameters, self.output_parameters = self.sort_parameters(arr)
        self.output_pi_groups = None
        self.input_pi_groups = None

    def sort_parameters(self, arr):
        free_param, input_param, output_param = [], [], []
        for i, param in enumerate(self.parameters):
            if arr[i] == 0:
                free_param.append(param)
            elif arr[i] == 1:
                input_param.append(param)
            elif arr[i] == 2:
                output_param.append(param)
        return free_param, input_param, output_param

    def _define_pi_groups(self):
        for i in range(len(self.parameters)):
            groups = combinations(self.parameters, i+1)
            for group in groups:
                print([str(a) for a in group])
                M = DimensionalMatrix(group)
                print(M)
                print('det:', M.det)
                # while M.det != 0:
                #     pass
                print(group)
                test = M
                print(test)
                a, b = test[:, 1:], test[:, 0]
                print(a, b)

    @staticmethod
    def find_repeating_variables(parameters):
        repeating_variables = []
        dimensional_matrix = DimensionalMatrix(parameters)
        # combinations = Util.combinations
        for group in combinations(parameters, dimensional_matrix.rank):
            M = DimensionalMatrix(group)
            if M.rank == dimensional_matrix.rank:
                repeating_variables.append(ListOfParameters(group))
        return repeating_variables


class ParameterType:
    free_param = 0
    input_param = 1
    output_param = 2


if __name__ == '__main__':

    d = Parameter(value=np.array([1, 1, 1]), units=Units.length, name='d')
    t = Parameter(value=np.array([1, 1, 1]), units=Units.time, name='t')
    v = Parameter(value=np.array([1, 1, 1]), units=Units.velocity, name='v')
    h = Parameter(value=np.array([1, 1, 1]), units=Units.length, name='h')
    A = Parameter(value=np.array([1, 1, 1]), units=Units.acceleration, name='A')

    u = Parameter(value=np.array([1, 1, 1]), units=Units.density, name='u')
    U = Parameter(value=np.array([1, 1, 1]), units=Units.surface_tension, name='U')
    y = Parameter(value=np.array([1, 1, 1]), units=Units.viscosity_dynamic, name='y')

    problem = ListOfParameters([u, U, y, d])
    solution = DimensionalAnalysis(problem)
    solution._define_pi_groups()

    # print(DimensionalMatrix([u, U, y]))
    # print(y.units, U.units)
    # test = DimensionalMatrix([d, v, U, u])
    # print(test)
    # a, b = test[:, 1:], test[:, 0]
    # print(a, b)
    # print(np.linalg.solve(a,b))
    # density, surface_tension, gravity, viscosity (kinematic/dynamic),
