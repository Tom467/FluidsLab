import sys
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from units import Units
from parameter import Parameter, ListOfParameters
from data_reader import Data
from buckingham_pi_theorem.dimensional_analysis import DimensionalAnalysis
from ui.available_parameters_ui import AvailableParameters


class ParameterCreator(QWidget):
    def __init__(self, parent=None):
        super(ParameterCreator, self).__init__(parent)
        self.available_parameters = ListOfParameters([])
        self.name = ''
        self.units = None
        self.values = None
        self.parameter = None
        self.file_location = None

        layout = QFormLayout()

        self.workspace = QVBoxLayout()
        for parameter in self.available_parameters:
            label = QLabel()
            label.setText(parameter.name)
            self.workspace.addWidget(label)
        layout.addRow('Workspace', self.workspace)

        self.parameter_name = QLineEdit()
        self.parameter_name.textChanged.connect(self.new_name)
        layout.addRow('Parameter Name', self.parameter_name)

        self.unit_options = QComboBox()
        self.unit_options.addItems(Units().get_units())
        self.unit_options.currentIndexChanged.connect(self.selection_change)
        layout.addRow('Units of Parameter', self.unit_options)

        self.av = QLineEdit()
        self.av.textChanged.connect(self.add_values)
        layout.addRow('Values', self.av)

        self.create_parameter = QPushButton()
        self.create_parameter.setText("Create Parameter")
        # self.b1.move(50, 20)
        self.create_parameter.clicked.connect(self.create)
        layout.addRow(self.create_parameter)

        self.browse = QPushButton("Browse Files")
        self.browse.clicked.connect(self.getfile)
        layout.addRow('Load Parameters', self.browse)

        self.analyze = QPushButton("Run Dimensional Analysis")
        self.analyze.clicked.connect(self.dimensional_analysis)
        layout.addRow('Analyze Parameters', self.analyze)

        self.setLayout(layout)
        self.setWindowTitle("Splash Lab")

    def new_name(self, text):
        self.name = text

    def selection_change(self, i):
        units = getattr(Units(), self.unit_options.currentText())
        self.units = units  # f'Current index {i} selection changed {self.cb.currentText()}'

    def add_values(self, text):
        self.values = text.split(',')

    def create(self):
        parameter = Parameter(value=[float(number) for number in self.values], units=self.units, name=self.name)
        self.available_parameters.append(parameter)
        label = QLabel()
        label.setText(f'{parameter.name} - {parameter.units}')
        self.workspace.addWidget(label)

    def getfile(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Data files (*.csv)")
        if fname == ('', ''):
            return
        self.file_location = fname[0]
        data = Data(self.file_location)
        for parameter in data.parameters:
            self.available_parameters.append(parameter)
        # for parameter in self.available_parameters:
            print(parameter.name)
            label = QLabel()
            label.setText(f'{parameter.name} - {parameter.units}')
            self.workspace.addWidget(label)
        # self.workspace.reset_layout()
        # self.reset_layout()

    def dimensional_analysis(self):
        if len(self.available_parameters) > 0:
            analysis = DimensionalAnalysis(self.available_parameters)


def main():
    app = QApplication(sys.argv)
    ex = ParameterCreator()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
    # units = Units()
    # attributes = [name for name in dir(units) if '__' not in name]
    # print(attributes)
