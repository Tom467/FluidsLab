import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from parameter import ListOfParameters, Parameter
from units import Units


class AvailableParameters(QWidget):
    def __init__(self, parent=None):
        super(AvailableParameters, self).__init__(parent)
        self.available_parameters = ListOfParameters([Parameter(value=2, units=Units.velocity, name='test')])
        self.reset_layout()

    def reset_layout(self):
        layout = QVBoxLayout()
        print('testing', self.available_parameters)
        for parameter in self.available_parameters:
            print(parameter.name)
            label = QLabel()
            label.setText(parameter.name)
            layout.addWidget(label)

        self.setLayout(layout)


def main():
    app = QApplication(sys.argv)
    ex = AvailableParameters()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
