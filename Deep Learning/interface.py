import sys

from PyQt5.QtWidgets import *
from PyQt5 import uic
from machine import main


class Window(QMainWindow):
    def __init__(self):
        super(Window, self).__init__()

        uic.loadUi('interface.ui', self)

        self.pushButton.clicked.connect(self.click)

    def click(self):
        f1, f2, f3, f4 = False, False, False, False
        if self.lineEdit.text().isdigit():
            if 1 <= int(self.lineEdit.text().isdigit()) <= 15:
                f1 = True
        if self.lineEdit_2.text().isdigit():
            if 2 <= int(self.lineEdit_2.text()) <= 5:
                f2 = True
        if self.lineEdit_3.text().isdigit():
            if 1 <= int(self.lineEdit_3.text()) <= 10:
                f3 = True
        if self.lineEdit_4.text() == "да":
            n = 1
            f4 = True
        if self.lineEdit_4.text() == "нет":
            n = 0
            f4 = True

        if f1 and f2 and f3 and f4:
            n1, n2, n3, n4 = int(self.lineEdit.text()), int(self.lineEdit_2.text()), int(self.lineEdit_3.text()), n
            text = main(n1, n2, n3, n4)
            self.label_5.setText(f"Скорей всего вы учитесь: {text}")


def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)
