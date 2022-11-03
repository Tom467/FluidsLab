import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from units import Units
from parameter import Parameter
from ui.available_parameters_ui import AvailableParameters


class ParameterCreator(QWidget):
    def __init__(self, parent=None):
        super(ParameterCreator, self).__init__(parent)
        self.workspace = AvailableParameters()
        self.name = ''
        self.units = None
        self.values = None
        self.parameter = None
        layout = QFormLayout()

        layout.addWidget(self.workspace)

        self.parameter_name = QLineEdit()
        self.parameter_name.textChanged.connect(self.new_name)
        layout.addRow('Parameter Name', self.parameter_name)

        self.cb = QComboBox()
        self.cb.addItems(Units().get_units())
        self.cb.currentIndexChanged.connect(self.selection_change)
        layout.addRow('Units of Parameter', self.cb)

        self.av = QLineEdit()
        self.av.textChanged.connect(self.add_values)
        layout.addRow('Values', self.av)

        self.b1 = QPushButton()
        self.b1.setText("Create Parameter")
        # self.b1.move(50, 20)
        self.b1.clicked.connect(self.b1_clicked)
        layout.addRow(self.b1)

        self.setLayout(layout)

    def new_name(self, text):
        self.name = text

    def selection_change(self, i):
        units = getattr(Units(), self.cb.currentText())
        self.units = units  # f'Current index {i} selection changed {self.cb.currentText()}'

    def add_values(self, text):
        self.values = text.split(',')

    def b1_clicked(self):
        self.parameter = Parameter(value=self.values, units=self.units, name=self.name)
        self.workspace.available_parameters.append(self.parameter)
        self.workspace.reset_layout()
        print(self.parameter.name, self.parameter.units, self.parameter.value)


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
