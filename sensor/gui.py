"""
提供GUI界面
"""
import tkinter as tk
from enum import Enum, unique

import PySimpleGUI as sg
import matplotlib.backends.tkagg as tkagg
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasAgg

from sensor.algorithm import AlgorithmManager
from sensor.algorithm import CycleDetectResult
from sensor.plot import PlotManager
from sensor.sensor import SensorManager
from settings import SENSOR_DATA


class GuiManager:
    @unique
    class KEYS(Enum):
        CANVAS_RAW_DATA = "用于显示原始数据的区域"
        CANVAS_GAIT_ACC = "显示加速度步态数据的区域"
        CANVAS_GEI_ACC = "显示加速度GEI的区域"
        CANVAS_GAIT_GYRO = "显示陀螺仪步态数据的区域"
        CANVAS_GEI_GYRO = "显示陀螺仪GEI的区域"
        CANVAS_STABILITY = "步态稳定性图"
        IMAGE_STATUS = "当前运动状态的图片"
        TEXT_ACTIVITY = "动作识别结果"
        TEXT_WHO_YOU_ARE = "身份识别结果"
        TEXT_IS_WALK_LIKE_DATA0 = "当前是否像data0一样"
        TEXT_CYCLE_DETECT_HISTORY = "步态周期的检测历史"
        TEXT_ACC_CYCLE_FEATURE = "加速度周期的特征"
        TEXT_GYRO_CYCLE_FEATURE = "陀螺仪周期的特征"

    def __init__(self):
        # gui通用设置
        sg.SetOptions(background_color="#FFFFFF", element_background_color="#FFFFFF", text_color="#000000")
        # plot manager，用于获取绘图信息
        self.sensor_manager = SensorManager(SENSOR_DATA)
        self.algorithm_manager = AlgorithmManager(self.sensor_manager)
        self.plot_manager = PlotManager(self.sensor_manager, self.algorithm_manager)
        self.text_init_placeholder = " " * 100

        # 构建gui
        self.layout = [
            [
                sg.Column([
                    [sg.Frame("原始数据", [
                        [sg.Canvas(size=(self.plot_manager.fig_raw_data.width, self.plot_manager.fig_raw_data.height),
                                   key=self.KEYS.CANVAS_RAW_DATA)]
                    ])]
                ]),
                sg.Column([
                    [sg.Frame("加速度步态", [
                        [sg.Canvas(size=(
                            self.plot_manager.fig_gait_acc.fig_width,
                            self.plot_manager.fig_gait_acc.fig_height),
                            key=self.KEYS.CANVAS_GAIT_ACC)],
                        [sg.Canvas(size=(
                            self.plot_manager.fig_gait_acc.fig_width,
                            self.plot_manager.fig_gait_acc.fig_height),
                            key=self.KEYS.CANVAS_GEI_ACC)],
                        [sg.Text(text=self.algorithm_manager.acc_data_pre_process.get_cycle_feature_for_gui(), key=self.KEYS.TEXT_ACC_CYCLE_FEATURE)]])
                     ],
                    [sg.Frame("陀螺仪步态", [
                        [sg.Canvas(size=(
                            self.plot_manager.fig_gait_gyro.fig_width,
                            self.plot_manager.fig_gait_gyro.fig_height),
                            key=self.KEYS.CANVAS_GAIT_GYRO)],
                        [sg.Canvas(size=(
                            self.plot_manager.fig_gait_gyro.fig_width,
                            self.plot_manager.fig_gait_gyro.fig_height),
                            key=self.KEYS.CANVAS_GEI_GYRO)],
                        [sg.Text(text=self.algorithm_manager.gyro_data_pre_process.get_cycle_feature_for_gui(), key=self.KEYS.TEXT_GYRO_CYCLE_FEATURE)]])
                     ]
                ]),
                sg.Column([
                    [sg.Frame("步态稳定性", [
                        [sg.Canvas(size=(
                            self.plot_manager.fig_stability.fig_width,
                            self.plot_manager.fig_stability.fig_height),
                            key=self.KEYS.CANVAS_STABILITY
                        )]
                    ])],
                    [sg.Frame("步行检测", [
                        [sg.Text(text="", key=self.KEYS.TEXT_IS_WALK_LIKE_DATA0)],
                    ])],
                    [sg.Frame("身份识别结果", [
                        [sg.Text(text="", key=self.KEYS.TEXT_WHO_YOU_ARE)],
                    ])],
                    [sg.Frame("动作识别结果", [
                        [sg.Text(text="", key=self.KEYS.TEXT_ACTIVITY)],
                    ])],
                    [sg.Frame("步态周期历史", [
                        [sg.Text(text=" " * 100, key=self.KEYS.TEXT_CYCLE_DETECT_HISTORY)]  # 100是为了搞个长的长度，不然显示不全
                    ])]
                ])
            ],
        ]
        self.window = sg.Window("demo").Layout(self.layout).Finalize()

    def _get_element(self, key):
        """
        获取GUI上的一个元素，不然太多的self.window.FindElement()
        :return:
        """
        return self.window.FindElement(key)

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

    @staticmethod
    def _update_gei_pic(gei_canvas, gei):
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
        self.plot_manager.update_display_raw_data_fig()
        # 更新步态周期图像
        self.plot_manager.update_gait_figure()
        # 步态稳定性
        self.plot_manager.fig_stability.update()

    def update_gui(self):
        """
        更新GUI。 注意：这里生成的各种乱七八糟的photo都要return回去，不然无法显示
        :return:
        """
        # 更新原始display图像
        raw_data_pic = self._plot_pic(self.plot_manager.fig_raw_data.fig,
                                      self._get_element(self.KEYS.CANVAS_RAW_DATA).TKCanvas)
        # 当前是否在步行
        self._get_element(self.KEYS.TEXT_IS_WALK_LIKE_DATA0).Update(value=self.algorithm_manager.is_walking)
        # 步态周期图
        acc_gait_and_gei_pic = self._update_gait_and_gei(self.plot_manager.fig_gait_acc.fig,
                                                         self._get_element(self.KEYS.CANVAS_GAIT_ACC).TKCanvas,
                                                         self._get_element(self.KEYS.CANVAS_GEI_ACC).TKCanvas,
                                                         self.plot_manager.fig_gait_acc.gei)
        gyro_gait_and_gei_pic = self._update_gait_and_gei(self.plot_manager.fig_gait_gyro.fig,
                                                          self._get_element(self.KEYS.CANVAS_GAIT_GYRO).TKCanvas,
                                                          self._get_element(self.KEYS.CANVAS_GEI_GYRO).TKCanvas,
                                                          self.plot_manager.fig_gait_gyro.gei)
        # 身份识别
        self._get_element(self.KEYS.TEXT_WHO_YOU_ARE).Update(value=self.algorithm_manager.who_you_are)
        gui_gait_stability = self._plot_pic(self.plot_manager.fig_stability.fig,
                                            self._get_element(self.KEYS.CANVAS_STABILITY).TKCanvas)
        self._get_element(self.KEYS.TEXT_CYCLE_DETECT_HISTORY).Update(value=" ".join(["{0}:{1}".format(
            cycle_detect_result.value[0], self.algorithm_manager.cycle_detect_history[cycle_detect_result])
            for cycle_detect_result in
            CycleDetectResult]))
        if self.algorithm_manager.acc_data_pre_process.last_cycle is not None:
            self._get_element(self.KEYS.TEXT_ACC_CYCLE_FEATURE) \
                .Update(value=self.algorithm_manager.acc_data_pre_process.get_cycle_feature_for_gui())
        if self.algorithm_manager.gyro_data_pre_process.last_cycle is not None:
            self._get_element(self.KEYS.TEXT_GYRO_CYCLE_FEATURE) \
                .Update(value=self.algorithm_manager.gyro_data_pre_process.get_cycle_feature_for_gui())
        return raw_data_pic, acc_gait_and_gei_pic, gyro_gait_and_gei_pic, gui_gait_stability

    def run(self):
        while True:
            event, values = self.window.Read(timeout=5)
            if not event:
                break
            self.update_data()
            self.update_fig()
            gui = self.update_gui()
