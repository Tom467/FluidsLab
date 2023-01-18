import numpy as np

from util import Util
from units import Units
from itertools import combinations
from parameter import ListOfParameters, Parameter
from buckingham_pi_theorem.pi_group import PiGroup
from buckingham_pi_theorem.dimensional_matrix import DimensionalMatrix


class DimensionalAnalysis:
    def __init__(self, parameters):
        self.parameters = ListOfParameters(parameters)
        self.repeating_variables = self.find_repeating_variables(parameters)

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

    def create_pi_groups(self, parameter):
        pi_groups = []
        for repeating_variable in self.repeating_variables:
            if parameter not in repeating_variable:
                pi_group = PiGroup(ListOfParameters([parameter]) + repeating_variable)
                if pi_group not in pi_groups:
                    pi_groups.append(pi_group)
        return pi_groups


if __name__ == '__main__':

    d = Parameter(value=np.array([1, 1, 1]), units=Units.length, name='d')
    t = Parameter(value=np.array([1, 1, 1]), units=Units.time, name='t')
    v = Parameter(value=np.array([1, 1, 1]), units=Units.velocity, name='v')
    h = Parameter(value=np.array([1, 1, 1]), units=Units.length, name='h')
    A = Parameter(value=np.array([1, 1, 1]), units=Units.acceleration, name='A')

    u = Parameter(value=np.array([1, 1, 1]), units=Units.density, name='u')
    U = Parameter(value=np.array([1, 1, 1]), units=Units.surface_tension, name='U')
    y = Parameter(value=np.array([1, 1, 1]), units=Units.viscosity_dynamic, name='y')

    problem = ListOfParameters([d, t, v, h, A, u, U, y])
    solution = DimensionalAnalysis(problem)
    A_pi_groups = solution.create_pi_groups(A)
    print(len(A_pi_groups))
    print(Util.list_to_string(A_pi_groups, newline=True))


    # print(DimensionalMatrix([u, U, y]))
    # print(y.units, U.units)
    # test = DimensionalMatrix([d, v, U, u])
    # print(test)
    # a, b = test[:, 1:], test[:, 0]
    # print(a, b)
    # print(np.linalg.solve(a,b))
    # density, surface_tension, gravity, viscosity (kinematic/dynamic),
