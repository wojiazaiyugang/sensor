"""
提供GUI界面
"""
import tkinter as tk
from enum import Enum, unique

import numpy
import PySimpleGUI as sg
from PIL import Image, ImageTk

import matplotlib.backends.tkagg as tkagg

from matplotlib.backends.backend_tkagg import FigureCanvasAgg

from sensor.algorithm import AlgorithmManager
from sensor.plot import PlotManager
from sensor.sensor import SensorManager
from util import get_static_file_full_path


class GuiManager:
    @unique
    class KEYS(Enum):
        CANVAS_RAW_DATA = "用于显示原始数据的区域"
        CANVAS_GAIT_ACC = "显示加速度步态数据的区域"
        CANVAS_GEI_ACC = "显示加速度GEI的区域"
        CANVAS_GAIT_GYRO = "显示陀螺仪步态数据的区域"
        CANVAS_GEI_GYRO = "显示陀螺仪GEI的区域"
        CANVAS_GAIT_ANG = "显示欧拉角步态数据的区域"
        CANVAS_GEI_ANG = "显示欧拉角GEI的区域"
        IMAGE_STATUS = "当前运动状态的图片"
        TEXT_ACTIVITY = "当前运动状态"
        TEXT_IS_WALK_LIKE_DATA0 = "当前是否像data0一样"

    def __init__(self):
        # gui通用设置
        sg.SetOptions(background_color="#FFFFFF", element_background_color="#FFFFFF", text_color="#FF0000")
        # plot manager，用于获取绘图信息
        self.sensor_manager = SensorManager(0)
        self.algorithm_manager = AlgorithmManager(self.sensor_manager)
        self.plot_manager = PlotManager(self.sensor_manager, self.algorithm_manager)

        # 构建gui
        self.layout = [
            [
                sg.Column([
                    [sg.Frame("原始数据", [
                        [sg.Canvas(size=(self.plot_manager.raw_data_fig.width, self.plot_manager.raw_data_fig.height),
                                   key=self.KEYS.CANVAS_RAW_DATA)]
                    ])]
                ]),
                sg.Column([
                    [sg.Frame("加速度步态", [
                        [sg.Canvas(size=(self.plot_manager.gait_acc_fig.fig_gait_acc_w, self.plot_manager.gait_acc_fig.fig_gait_acc_h),
                                   key=self.KEYS.CANVAS_GAIT_ACC)],
                        [sg.Canvas(size=(self.plot_manager.gait_acc_fig.fig_gait_acc_w, self.plot_manager.gait_acc_fig.fig_gait_acc_h),
                                   key=self.KEYS.CANVAS_GEI_ACC)]])
                     ],
                    [sg.Frame("陀螺仪步态", [
                        [sg.Canvas(size=(self.plot_manager.gait_gyro_fig.fig_gait_acc_w,self.plot_manager.gait_gyro_fig.fig_gait_acc_h),
                                   key=self.KEYS.CANVAS_GAIT_GYRO)],
                        [sg.Canvas(size=(self.plot_manager.gait_gyro_fig.fig_gait_acc_w,self.plot_manager.gait_gyro_fig.fig_gait_acc_h),
                                   key=self.KEYS.CANVAS_GEI_GYRO)]])
                     ],
                    [sg.Frame("欧拉角步态", [
                        [sg.Canvas(size=(self.plot_manager.gait_ang_fig.fig_gait_acc_w, self.plot_manager.gait_ang_fig.fig_gait_acc_h),
                                   key=self.KEYS.CANVAS_GAIT_ANG)],
                        [sg.Canvas(size=(self.plot_manager.gait_ang_fig.fig_gait_acc_w, self.plot_manager.gait_ang_fig.fig_gait_acc_h),
                                   key=self.KEYS.CANVAS_GEI_ANG)]])
                     ],
                ]),
                sg.Column([
                    [sg.Image(filename=get_static_file_full_path("1.png"), key=self.KEYS.IMAGE_STATUS,
                              size=(100, 100))],
                    [sg.Text(text="日志", key=self.KEYS.TEXT_ACTIVITY)],
                    [sg.Text(text="data0", key=self.KEYS.TEXT_IS_WALK_LIKE_DATA0)]
                ])
            ],
            # [
            #     sg.Output(size=(200,5))
            # ]
        ]
        self.window = sg.Window("demo").Layout(self.layout).Finalize()

    def _update_gait_pic(self, figure, gait_canvas):
        """
        在pysimplegui上绘制plot。调用这个函数必须接受返回值，不接受的话无法绘图，我也不知道为啥，辣鸡tkinter
        :param gait_canvas:
        :param figure:
        :return:
        """
        figure_canvas_agg = FigureCanvasAgg(figure)
        figure_canvas_agg.draw()
        figure_x, figure_y, figure_w, figure_h = figure.bbox.bounds
        figure_w, figure_h = int(figure_w), int(figure_h)
        photo = tk.PhotoImage(master=gait_canvas, width=figure_w, height=figure_h)
        gait_canvas.create_image(figure_w / 2, figure_h / 2, image=photo)
        tkagg.blit(photo, figure_canvas_agg.get_renderer()._renderer, colormode=2)
        return photo

    def _update_gei_pic(self, fig, gei_canvas, gei):
        """
        更新gei图像
        :return:
        """
        if gei is None:
            return None
        figure_photo_gei = ImageTk.PhotoImage(image=Image.fromarray(gei))
        gei_canvas.create_image(0, 0, image=figure_photo_gei,anchor=tk.NW)
        return figure_photo_gei

    def _update_gait_and_gei(self, fig, gait_canvas, gei_canvas, gei):
        """
        同时更新gait和gei
        :param fig:
        :param gait_canvas:
        :return:
        """
        gait = self._update_gait_pic(fig, gait_canvas)
        gei = self._update_gei_pic(fig, gei_canvas, gei)
        return gait, gei

    def run(self):
        while True:
            event, values = self.window.Read(timeout=5)
            if not event:
                break
            self.plot_manager.update_figures()
            raw_data_pic = self._update_gait_pic(self.plot_manager.raw_data_fig.fig, self.window.FindElement(self.KEYS.CANVAS_RAW_DATA).TKCanvas,)
            acc = self._update_gait_and_gei(self.plot_manager.gait_acc_fig.fig,
                                            self.window.FindElement(self.KEYS.CANVAS_GAIT_ACC).TKCanvas,
                                            self.window.FindElement(self.KEYS.CANVAS_GEI_ACC).TKCanvas, self.plot_manager.gait_acc_fig.get_gei())

            gyro = self._update_gait_and_gei(self.plot_manager.gait_gyro_fig.fig,
                                             self.window.FindElement(self.KEYS.CANVAS_GAIT_GYRO).TKCanvas,
                                             self.window.FindElement(self.KEYS.CANVAS_GEI_GYRO).TKCanvas, self.plot_manager.gait_gyro_fig.get_gei())

            ang = self._update_gait_and_gei(self.plot_manager.gait_ang_fig.fig,
                                             self.window.FindElement(self.KEYS.CANVAS_GAIT_ANG).TKCanvas,
                                             self.window.FindElement(self.KEYS.CANVAS_GEI_ANG).TKCanvas, self.plot_manager.gait_ang_fig.get_gei())
            # # 更新加速度步态图像
            # figure_photo_gait_acc = self._draw_figure(self.window.FindElement(self.KEYS.CANVAS_GAIT_ACC).TKCanvas,
            #                                           self.plot_manager.fig_acc_gait)
            # # 更新gei图像
            # gei_pic = self._update_gei_pic(self.plot_manager.fig_acc_gait)
            # # 更新陀螺仪步态图像
            # figure_photo_gait_gyro = self._draw_figure(self.window.FindElement(self.KEYS.CANVAS_GAIT_GYRO).TKCanvas,
            #                                            self.plot_manager.fig_gyro_gait)

            # 更新当前运动状态图
            # predict_number = self.algorithm_manager.get_current_activity()
            # self.window.FindElement(self.KEYS.TEXT_ACTIVITY).Update(
            #     value=self.algorithm_manager.activity_recognition_network.REVERSED_LABEL_MAP.get(predict_number))
            # if predict_number == 1 or predict_number == 2:
            #     status = "sit"
            # elif predict_number == 3:
            #     status = "walk"
            # else:
            #     status = "run"
            # self.window.FindElement(self.KEYS.IMAGE_STATUS).UpdateAnimation(
            #     get_static_file_full_path("{0}.gif".format(status)))
            # self.window.FindElement(self.KEYS.TEXT_IS_WALK_LIKE_DATA0).Update(value = self.algorithm_manager.is_walk_like_data0())
