import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from ui.parameter_creator_ui import ParameterCreator


def window():
    app = QApplication(sys.argv)
    win = QWidget()
    layout = QHBoxLayout()

    menu = ParameterCreator()
    menu.move(50, 40)
    layout.addWidget(menu)

    win.setLayout(layout)
    # win.resize(900, 800)

    win.setWindowTitle("Fluids Lab")
    win.show()
    sys.exit(app.exec_())


def b1_clicked():
    print("Button 1 clicked")


def b2_clicked():
    print("Button 2 clicked")


if __name__ == '__main__':
    window()
