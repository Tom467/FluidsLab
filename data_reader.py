import pandas as pd
from units import Units
from convert import Convert, ConvertTemperature
from parameter import Parameter, ListOfParameters
from buckingham_pi_theorem.dimensional_analysis import DimensionalAnalysis


class Data:
    def __init__(self, file):
        self.file_location = '' if isinstance(file, pd.core.frame.DataFrame) else file
        self.data = file if isinstance(file, pd.core.frame.DataFrame) else self.read_file(self.file_location)
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
    experiment = Data("C:/Users/truma/Downloads/testdata3.csv")
    print([param.name for param in experiment.parameters[2:6]])
    print([param.name for param in experiment.parameters[2:4]])
    d = DimensionalAnalysis(experiment.parameters[2:7], repeating_parameters=experiment.parameters[2:4])
    d.plot()

#['V_i', 'd_{hmax}', 'a', 'b', 'g'] ['V_i', 'd_{hmax}']

    # [print(group, '\n', group.repeating_variables) for group in d.pi_group_sets]

    # values = [80, 20, 9.8, 1, 1, 1]
    # test = ListOfParameters([])
    # for i, parameter in enumerate(experiment.parameters[1:]):
    #     # print(Parameter(value=values[i], units=parameter.units, name=parameter.name))
    #     test.append(Parameter(value=values[i], units=parameter.units, name=parameter.name))
    # print('test', test)
    # test = d.predict(experiment.parameters)
