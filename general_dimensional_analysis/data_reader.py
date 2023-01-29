
import numpy as np
import pandas as pd
from general_dimensional_analysis.unit import Units
from general_dimensional_analysis.parameter import Parameter
from general_dimensional_analysis.group_of_parameter import GroupOfParameters

# from buckingham_pi_theorem.dimensional_analysis import DimensionalAnalysis


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
        parameters = []
        for key in self.data:
            parameters.append(Parameter(values=np.array([value for value in self.data[key]]).astype(np.float64),
                                        units=getattr(Units(), key.split('-')[1]),
                                        name=key.split('-')[0]))
        return GroupOfParameters(parameters)


if __name__ == "__main__":
    experiment = Data("C:/Users/truma/Downloads/testdata3.csv")
    print(experiment.parameters)
    # d = DimensionalAnalysis(experiment.parameters[2:7], repeating_parameters=experiment.parameters[2:4])
    # d.plot()
