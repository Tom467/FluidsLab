import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from ui.parameter_creator_ui import ParameterCreator
from ui.file_reader_ui import FileReader


class WorkspaceUI(QWidget):
    def __init__(self, parent=None):
        super(WorkspaceUI, self).__init__(parent)

        layout = QVBoxLayout()

        self.new_parameters = ParameterCreator()
        layout.addWidget(self.new_parameters)

        self.loaded_parameters = FileReader()
        layout.addWidget(self.loaded_parameters)

        self.setLayout(layout)


def main():
    app = QApplication(sys.argv)
    ex = WorkspaceUI()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
