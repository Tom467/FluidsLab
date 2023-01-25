import numpy as np
import matplotlib.pyplot as plt

from util import Util
from units import Units
from data_reader import Data
from itertools import combinations
from parameter import ListOfParameters, Parameter
from buckingham_pi_theorem.pi_group import PiGroup
from buckingham_pi_theorem.dimensional_matrix import DimensionalMatrix


class DimensionalAnalysis:
    def __init__(self, parameters: ListOfParameters):
        self.parameters = ListOfParameters(parameters)
        self.dimensional_matrix = DimensionalMatrix(self.parameters)
        self.repeating_variables = self.find_repeating_variables(parameters)

    def __iter__(self):
        for elem in self.parameters:
            yield elem

    def __add__(self, other: Parameter):
        return DimensionalAnalysis(self.parameters + other)

    @staticmethod
    def find_repeating_variables(parameters):
        repeating_variables = []
        dimensional_matrix = DimensionalMatrix(parameters)
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

    @staticmethod
    def plot(x, y):
        plt.scatter(x.values, y.values)
        plt.xlabel(r'$a$'.replace('a', x.name) if isinstance(x, Parameter) else x.formula)
        plt.ylabel(r'$a$'.replace('a', y.name) if isinstance(y, Parameter) else y.formula)
        plt.show()


if __name__ == '__main__':

    d = Parameter(value=np.array([1, 1, 1]), units=Units.length, name='d')
    t = Parameter(value=np.array([1, 1, 1]), units=Units.time, name='t')
    v = Parameter(value=np.array([1, 1, 1]), units=Units.velocity, name='v')
    h = Parameter(value=np.array([1, 1, 1]), units=Units.length, name='h')
    A = Parameter(value=np.array([1, 1, 1]), units=Units.acceleration, name='A')

    u = Parameter(value=np.array([1, 1, 1]), units=Units.density, name='u')
    U = Parameter(value=np.array([1, 1, 1]), units=Units.surface_tension, name='U')
    y = Parameter(value=np.array([1, 1, 1]), units=Units.viscosity_dynamic, name='y')

    problem = ListOfParameters([d, t, v])  # , h, A, u, U, y])
    print('given', problem)
    print('problem', problem + u)
    solution = DimensionalAnalysis(problem - v + y)
    print(solution.dimensional_matrix)

    # print('plotting')
    # experiment = Data("C:/Users/truma/Downloads/testdata3.csv")
    # analysis = DimensionalAnalysis(experiment.parameters)
    # DimensionalAnalysis.plot(experiment.parameters[0]**2, analysis.create_pi_groups(analyis.parameters[4]*experiment.parameters[1])[0])

