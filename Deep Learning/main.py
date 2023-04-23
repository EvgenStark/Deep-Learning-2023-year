import sys

from interface import Window, except_hook
from PyQt5.QtWidgets import QApplication


# 5 5 8 да -> отлично
# 2 5 4 да -> хорошо
# 8 3 10 да -> хорошо


if __name__ == "__main__":
    print("Приветсвуем вас в нашей нейронной сети!")

    # входных нейронов 4
    # неройнов среднего слоя 20
    # выходных нейрона 3

    # точность нейронной сети 93%

    application = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.excepthook = except_hook
    sys.exit(application.exec())