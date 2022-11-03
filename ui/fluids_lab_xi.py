import sys
from parameter import ListOfParameters, Parameter
from buckingham_pi_theorem.dimensional_analysis import DimensionalAnalysis
from ui.ui_main import window
from PyQt5.QtWidgets import *


class Workspace:
    def __init__(self, parameters):  # parameters must be type ListOfParameters
        self.parameters = parameters
        self.show = window()


# def main():
#     app = QApplication(sys.argv)
#     ex = Window()
#     ex.show()
#     sys.exit(app.exec_())


if __name__ == '__main__':
    Workspace(ListOfParameters([])).show
