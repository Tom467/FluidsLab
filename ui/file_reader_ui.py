from data_reader import Data
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class FileReader(QWidget):
    def __init__(self, parent=None):
        super(FileReader, self).__init__(parent)
        self.file_location = None
        self.data = None
        self.reset_layout()

    def reset_layout(self):
        layout = QVBoxLayout()

        self.btn = QPushButton("Browse Files")
        self.btn.clicked.connect(self.getfile)
        layout.addWidget(self.btn)

        self.file = QLabel()
        layout.addWidget(self.file)
        self.setLayout(layout)
        self.setWindowTitle("Load CSV File")

    def getfile(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                            'c:\\', "Data files (*.csv)")
        self.file_location = fname[0]
        self.file.setText(self.file_location)
        self.data = Data(self.file_location)
        self.file.setText(str([param.name for param in self.data.parameters]))


def main():
    app = QApplication(sys.argv)
    ex = FileReader()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

