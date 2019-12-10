import matplotlib
# matplotlib.use('TkAgg')  # 兼容mac
import sys
from sensor.gui import GuiManager

if __name__ == "__main__":
    GuiManager().run()