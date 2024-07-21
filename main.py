import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFrame
from mainwindow import MainWidget  # Import the MainWidget class from your modified file

class AppFrame(QFrame):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Main Window Frame')
        self.setGeometry(100, 100, 800, 600)

        # Initialize MainWidget
        self.main_widget = MainWidget()

        # Layout
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.main_widget)
        self.setLayout(self.layout)

        self.show()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    frame = AppFrame()
    sys.exit(app.exec_())

