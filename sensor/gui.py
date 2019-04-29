import tkinter as tk
from enum import Enum, unique
import PySimpleGUI as sg
import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, FigureCanvasAgg

from sensor.plot import PlotManager


class GuiManager:
    @unique
    class KEYS(Enum):
        CANVAS = "用于显示数据的区域"

    def __init__(self):
        # plot manager，用于获取绘图信息
        self.plot_manager = PlotManager()

        # 构建gui
        self.layout = [
            [sg.Text("213213")],
            [sg.Canvas(size=(self.plot_manager.figure_w, self.plot_manager.figure_h), key=self.KEYS.CANVAS),sg.Text("213213")]
        ]

        self.ax = self.plot_manager.ax_acc
        self.window = sg.Window("demo").Layout(self.layout)
        self.window.Finalize()
        self.canvas_element = self.window.FindElement(self.KEYS.CANVAS)
        self.graph = FigureCanvasTkAgg(self.plot_manager.fig, master=self.canvas_element.TKCanvas)
        self.canvas = self.canvas_element.TKCanvas

    def run(self):
        while True:
            event, values = self.window.Read(timeout=1)
            if not event:
                break
            self.plot_manager.update()
            photo = tk.PhotoImage(master=self.canvas, width=self.plot_manager.figure_w, height=self.plot_manager.figure_h)
            self.canvas.create_image(self.plot_manager.figure_w/2, self.plot_manager.figure_h/2, image=photo)

            figure_canvas_agg = FigureCanvasAgg(self.plot_manager.fig)
            figure_canvas_agg.draw()
            tkagg.blit(photo, figure_canvas_agg.get_renderer()._renderer, colormode=2)
