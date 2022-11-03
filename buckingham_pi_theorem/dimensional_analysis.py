import copy
import numpy as np
import matplotlib.pyplot as plt

from util import Util
from units import Units
from parameter import Parameter, ListOfParameters
from buckingham_pi_theorem.pi_group import PiGroup
from machine_learning.gradient_descent import GradientDescent
from buckingham_pi_theorem.dimensional_matrix import DimensionalMatrix


class DimensionalAnalysis:
    def __init__(self, parameters, dependent_parameter=None):  # parameters should be of type ListOfParameter
        # TODO add logic to accept just a list of units list of parameters without values
        self.parameters = ListOfParameters(parameters)
        self.dependent_parameter = dependent_parameter
        self.units_of_parameters = self.parameters.units
        self.independent_dimensions = self.units_of_parameters.independent_dimensions
        self.number_of_pi_groups = self.calculate_number_of_pi_groups()
        self.dimensional_matrix = DimensionalMatrix(self.units_of_parameters)
        self.repeating_variables = self.find_repeating_variables()
        self.pi_groups = []
        self.create_pi_groups()
        self.regression_model = None
        self.generate_model()

    def predict(self, x):
        # values = []
        # for pi_group in self.pi_groups[1:]:
        #     x_temp = []
        #     for parameter in x:
        #         if pi_group.contains(parameter.name):
        #             x_temp.append(parameter)
        #     if len(x_temp) < 1:
        #         pass
        #     value = pi_group.calculate_value(x_temp)
        #     values.append(value)
        # values = np.array(values).T
        # print('shape', values.shape)
        values = np.array([copy.deepcopy(pi_group.value) for pi_group in self.pi_groups[1:]])
        return self.regression_model.predict(values.T)

    def generate_model(self):
        x = np.array([copy.deepcopy(pi_group.value) for pi_group in self.pi_groups[1:]])
        y = copy.deepcopy(self.pi_groups[0].value)
        self.regression_model = GradientDescent(np.transpose(x), y)

    def calculate_number_of_pi_groups(self):
        n = len(self.units_of_parameters)
        r = len(self.independent_dimensions)
        return n - r

    def find_repeating_variables(self):
        repeating_variables = []
        combinations = Util.combinations(self.parameters, self.dimensional_matrix.rank)
        for group in combinations:
            M = DimensionalMatrix(group)
            if M.rank == self.dimensional_matrix.rank:
                repeating_variables.append(group)
        return repeating_variables

    def create_pi_groups(self):
        group = self.parameters - self.repeating_variables[0]
        for variable in group:
            pi_group = PiGroup(ListOfParameters([variable]) + self.repeating_variables[0])
            self.pi_groups.append(pi_group)
        test = True
        if test:
            pass
        else:
            # The following loop can find oll the possible pi groups from all the different combinations of repeating variables
            for repeating_variables in self.repeating_variables:
                group = self.parameters - repeating_variables
                for variable in group:
                    pi_group = PiGroup(ListOfParameters([variable]) + repeating_variables)
                    self.pi_groups.append(pi_group)
                    # TODO the following if statement should not be needed
                    # if pi_group not in self.pi_groups:
                    #     self.pi_groups.append(pi_group)

    def plot(self):
        figure, axis = plt.subplots(len(self.pi_groups)-1, 1)
        y = self.pi_groups[0]
        for i, pi_group in enumerate(self.pi_groups[1:]):
            x = pi_group.value
            axis[i].scatter(x, y.value)
            axis[i].set_title(y.formula + ' vs. ' + pi_group.formula)

        self.regression_model.plot()
        plt.show()
        # index = range(len(y.value))
        # axis[i+1].scatter(index, y.value, s=10, c='b', marker="s", label='measured')
        # axis[i+1].scatter(index, self.predict(self.parameters), s=10, c='r', marker="o", label='predicted')
        # plt.legend(loc='upper left')
        # plt.show()


if __name__ == '__main__':

    # # Reynolds number
    rho = Parameter(value=1000, units=Units.density, name='rho')
    u = Parameter(value=[.15, 0.2], units=Units.velocity, name='u')
    hz = Parameter(value=np.array([10, 15, 20, 25]), units=Units.frequency, name='Hz')
    L1 = Parameter(value=np.array([0.025, 0.0125]), units=Units.length, name='L')
    mu = Parameter(value=8.9e-4, units=Units.viscosity_dynamic, name='mu')

    dP = Parameter(value=1000, units=Units.pressure, name='dP')
    U_ave = Parameter(value=1000, units=Units.velocity, name='U_ave')
    d1 = Parameter(value=1000, units=Units.length, name='d1')
    d2 = Parameter(value=1000, units=Units.length, name='d2')
    rho = Parameter(value=1000, units=Units.density, name='rho')
    mu = Parameter(value=8.9e-4, units=Units.viscosity_dynamic, name='mu')

    param = ListOfParameters([dP, U_ave, d1, d2, rho, mu])  # [Units.velocity, Units.density, Units.length, Units.time]
    D = DimensionalAnalysis(param)
    for group in D.pi_groups:
        print('pi group', group)
    # print(len(param))
    # D = DimensionalAnalysis(param, hz)
    # for group in D.pi_groups:
    #     print(str(group))
    # D.plot_pi_groups()
    # print([str(group) for group in D.repeating_variables])

    # group = ListOfParameters([rho, u, L1])
    # print(L2 not in group)
    # TODO test a list of parameters that are dimensionless then with groups of ranks 1-5
    # TODO test a dimensionless group that has fractional exponents
