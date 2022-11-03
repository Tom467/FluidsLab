import numpy as np
from buckingham_pi_theorem.dimensional_matrix import DimensionalMatrix


class PiGroup:
    def __init__(self, parameters):  # parameters should be of type ListOfParameters with the first parameter plus repeating variables
        self.parameters = parameters
        self.value = None
        self.exponents = None
        self._define_pi_group()
        self.formula = None
        self.formula_inverse = None
        self._define_formula()
        self.repeating_variables = parameters[1:]
        # TODO add some check to see if the Pi group is something common like the Reynold's Number

    def __str__(self):
        return str(self.value) + ' ' + str(self.formula)

    def __eq__(self, other):
        return self.formula == other.formula or self.formula == other.formula_inverse

    def _define_pi_group(self):
        M = DimensionalMatrix(self.parameters.units).M
        A, B = M[:, 1:], M[:, 0]
        self.exponents = np.round(-(np.linalg.inv(A) @ B), 2)
        self.value = self.calculate_value(self.parameters)
        # TODO add logic to make sure x is a vector of integers if raising units to this power

    def calculate_value(self, parameters):
        value = parameters[0].value
        for i, parameter in enumerate(parameters[1:]):
            value *= parameter.value**self.exponents[i]
        return value
        # TODO figure out what to return in addition to the total

    def contains(self, other_name):
        for param in self.parameters:
            if param.name == other_name:
                return True
        return False

    def _define_formula(self):
        top = ''
        bottom = ''
        for i, parameter in enumerate(self.parameters):
            if i == 0:
                top += f'({parameter.name})'
            else:
                if self.exponents[i-1] > 0:
                    if self.exponents[i-1] == 1:
                        top += f'({parameter.name})'
                    else:
                        top += f'({parameter.name})^{self.exponents[i-1]}'
                elif self.exponents[i-1] < 0:
                    if self.exponents[i-1] == -1:
                        bottom += f'({parameter.name})'
                    else:
                        bottom += f'({parameter.name})^{-self.exponents[i-1]}'

        self.formula = f'{top} / {bottom}' if bottom else top
        self.formula_inverse = f'{bottom} / {top}' if bottom else top
