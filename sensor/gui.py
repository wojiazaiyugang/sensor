"""
提供GUI界面
"""
import tkinter as tk
from enum import Enum, unique

import PySimpleGUI as sg
from PIL import Image, ImageTk
import matplotlib.backends.tkagg as tkagg
from matplotlib.backends.backend_tkagg import FigureCanvasAgg

from sensor.algorithm import AlgorithmManager
from sensor.plot import PlotManager
from sensor.sensor import SensorManager


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
        TEXT_ACTIVITY = "动作识别结果"
        TEXT_WHO_YOU_ARE = "身份识别结果"
        TEXT_IS_WALK_LIKE_DATA0 = "当前是否像data0一样"

    def __init__(self):
        # gui通用设置
        sg.SetOptions(background_color="#FFFFFF", element_background_color="#FFFFFF", text_color="#000000")
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
                        [sg.Canvas(size=(
                            self.plot_manager.gait_acc_fig.fig_gait_acc_w,
                            self.plot_manager.gait_acc_fig.fig_gait_acc_h),
                            key=self.KEYS.CANVAS_GAIT_ACC)],
                        [sg.Canvas(size=(
                            self.plot_manager.gait_acc_fig.fig_gait_acc_w,
                            self.plot_manager.gait_acc_fig.fig_gait_acc_h),
                            key=self.KEYS.CANVAS_GEI_ACC)]])
                     ],
                    [sg.Frame("陀螺仪步态", [
                        [sg.Canvas(size=(
                            self.plot_manager.gait_gyro_fig.fig_gait_acc_w,
                            self.plot_manager.gait_gyro_fig.fig_gait_acc_h),
                            key=self.KEYS.CANVAS_GAIT_GYRO)],
                        [sg.Canvas(size=(
                            self.plot_manager.gait_gyro_fig.fig_gait_acc_w,
                            self.plot_manager.gait_gyro_fig.fig_gait_acc_h),
                            key=self.KEYS.CANVAS_GEI_GYRO)]])
                     ],
                    [sg.Frame("欧拉角步态", [
                        [sg.Canvas(size=(
                            self.plot_manager.gait_ang_fig.fig_gait_acc_w,
                            self.plot_manager.gait_ang_fig.fig_gait_acc_h),
                            key=self.KEYS.CANVAS_GAIT_ANG)],
                        [sg.Canvas(size=(
                            self.plot_manager.gait_ang_fig.fig_gait_acc_w,
                            self.plot_manager.gait_ang_fig.fig_gait_acc_h),
                            key=self.KEYS.CANVAS_GEI_ANG)]])
                     ],
                ]),
                sg.Column([
                    [sg.Frame("步行检测", [
                        [sg.Text(text="", key=self.KEYS.TEXT_IS_WALK_LIKE_DATA0)],
                    ])],
                    [sg.Frame("身份识别结果", [
                        [sg.Text(text="", key=self.KEYS.TEXT_WHO_YOU_ARE)],
                    ])],
                    [sg.Frame("动作识别结果", [
                        [sg.Text(text="", key=self.KEYS.TEXT_ACTIVITY)],
                    ])],

                ])
            ],
            # [
            #     sg.Output(size=(200,5))
            # ]
        ]
        self.window = sg.Window("demo").Layout(self.layout).Finalize()

    def _plot_pic(self, figure, gait_canvas):
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

    def _update_gei_pic(self, gei_canvas, gei):
        """
        更新gei图像
        :return:
        """
        if gei is None:
            return None
        figure_photo_gei = ImageTk.PhotoImage(image=Image.fromarray(gei))
        gei_canvas.create_image(0, 0, image=figure_photo_gei, anchor=tk.NW)
        return figure_photo_gei

    def _update_gait_and_gei(self, fig, gait_canvas, gei_canvas, gei):
        """
        同时更新gait和gei
        :param fig:
        :param gait_canvas:
        :return:
        """
        gait = self._plot_pic(fig, gait_canvas)
        gei = self._update_gei_pic(gei_canvas, gei)
        return gait, gei

    def update_data(self):
        """
        更新程序中所有的数据
        :return:
        """
        # 更新原始display数据
        self.sensor_manager.update_display_raw_data()
        # 更新算法的所有结果
        self.algorithm_manager.update_data()

    def update_fig(self):
        """
        更新程序中所有plt显示的图
        :return:
        """
        # 更新原始display图像
        self.plot_manager.update_display_raw_data()
        # 更新步态周期图像
        self.plot_manager.gait_acc_fig.update_cycle_fig()
        self.plot_manager.gait_gyro_fig.update_cycle_fig()
        self.plot_manager.gait_ang_fig.update_cycle_fig()

    def update_gui(self):
        """
        更新GUI。 注意：这里生成的各种乱七八糟的photo都要return回去，不然无法显示
        :return:
        """
        # 更新原始display图像
        raw_data_pic = self._plot_pic(self.plot_manager.raw_data_fig.fig,
                                      self.window.FindElement(self.KEYS.CANVAS_RAW_DATA).TKCanvas)
        self.window.FindElement(self.KEYS.TEXT_IS_WALK_LIKE_DATA0).Update(value=self.algorithm_manager.is_walking)

        acc_gait_and_gei_pic = self._update_gait_and_gei(self.plot_manager.gait_acc_fig.fig,
                                        self.window.FindElement(self.KEYS.CANVAS_GAIT_ACC).TKCanvas,
                                        self.window.FindElement(self.KEYS.CANVAS_GEI_ACC).TKCanvas,
                                        self.plot_manager.gait_acc_fig.get_gei())
        gyro_gait_and_gei_pic = self._update_gait_and_gei(self.plot_manager.gait_gyro_fig.fig,
                                         self.window.FindElement(self.KEYS.CANVAS_GAIT_GYRO).TKCanvas,
                                         self.window.FindElement(self.KEYS.CANVAS_GEI_GYRO).TKCanvas,
                                         self.plot_manager.gait_gyro_fig.get_gei())

        ang_gait_and_gei_pic = self._update_gait_and_gei(self.plot_manager.gait_ang_fig.fig,
                                        self.window.FindElement(self.KEYS.CANVAS_GAIT_ANG).TKCanvas,
                                        self.window.FindElement(self.KEYS.CANVAS_GEI_ANG).TKCanvas,
                                        self.plot_manager.gait_ang_fig.get_gei())
        self.window.FindElement(self.KEYS.TEXT_WHO_YOU_ARE).Update(value=self.algorithm_manager.who_you_are)
        return raw_data_pic, acc_gait_and_gei_pic, gyro_gait_and_gei_pic, ang_gait_and_gei_pic

    def run(self):
        while True:
            event, values = self.window.Read(timeout=5)
            if not event:
                break
            self.update_data()
            self.update_fig()
            gui = self.update_gui()


