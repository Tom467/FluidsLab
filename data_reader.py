import pandas as pd
from units import Units
from convert import Convert, ConvertTemperature
from parameter import Parameter, ListOfParameters
from buckingham_pi_theorem.dimensional_analysis import DimensionalAnalysis


class Data:
    def __init__(self, file_location):
        self.file_location = file_location
        self.data = self.read_file(self.file_location)
        self.parameters = self.generate_list_of_parameters()

    @staticmethod
    def read_file(file_location):
        data = pd.read_csv(file_location)
        return data

    def generate_list_of_parameters(self):
        # TODO add the ability to convert to standard units (i.e. mm to m) using Convert and ConvertTemperature
        parameters = ListOfParameters([])
        for key in self.data:
            parameters.append(Parameter(value=[value for value in self.data[key]],
                                        units=getattr(Units(), key.split('-')[1]),
                                        name=key.split('-')[0]))
        return parameters


if __name__ == "__main__":
    experiment = Data("C:/Users/truma/Downloads/test - bernoulli.csv")
    print(experiment.parameters)
    d = DimensionalAnalysis(experiment.parameters)
    d.plot()
    print('pi groups', [group.formula for group in d.pi_groups])

    values = [80, 20, 9.8, 1, 1, 1]
    test = ListOfParameters([])
    for i, parameter in enumerate(experiment.parameters[1:]):
        # print(Parameter(value=values[i], units=parameter.units, name=parameter.name))
        test.append(Parameter(value=values[i], units=parameter.units, name=parameter.name))
    print('test', test)
    # test = d.predict(experiment.parameters)
