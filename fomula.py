from units import Units

# TODO this class needs to be written


class Formula:
    def __init__(self, formula):
        self.formula = formula
        variables = self._define_formula(formula)
        self.variables = {variable: 0 for variable in variables}

    def __str__(self):
        return self.formula

    def _define_formula(self, test):
        return test * self.formula
