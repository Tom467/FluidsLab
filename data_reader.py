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
            try:
                print(f"Processing key: {key}")  # Debugging statement
                parts = key.split('-')
                if len(parts) > 1 and parts[1]:  # Check if there's a valid second part
                    unit_key = parts[1]
                    unit = getattr(Units, unit_key, None)
                    if unit is not None:
                        parameters.append(Parameter(value=[value for value in self.data[key]],
                                                units=unit,
                                                name=parts[0]))
                    else:
                        print(f"Attribute '{unit_key}' not found in Units class")
                else:
                    print(f"Key '{key}' does not have a valid second part after hyphen, put into form parameter name-base unit")
            except Exception as e:
                print(f"Error processing key '{key}': {e}")
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
