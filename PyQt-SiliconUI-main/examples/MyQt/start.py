import os
import sys
import time
import qt5reactor
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
import siui
from siui.core import SiGlobal
# siui.gui.set_scale_factor(1)
from ui import MySiliconApp


if __name__ == "__main__":
    app = QApplication(sys.argv)
    if "twisted.internet.reactor" in sys.modules:
        del sys.modules["twisted.internet.reactor"]
    qt5reactor.install()

    window = MySiliconApp()
    window.show()

    timer = QTimer(window)
    sys.exit(app.exec_())
