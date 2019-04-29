from numpy import short
import pywinusb.hid as hid


class SensorManager:
    """
    传感器管理
    """

    def __init__(self):
        # 传感器
        self.sensor = self._get_sensor()
        # 最多保存的数据点的个数
        self.ACC_POINT_COUNT = 400
        self.GYRO_POINT_COUNT = 400
        self.ANG_POINT_COUNT = 400
        # 加速度数据
        self.acc_x, self.acc_y, self.acc_z = [], [], []
        # 陀螺仪数据
        self.gyro_x, self.gyro_y, self.gyro_z = [], [], []
        # 角度
        self.ang_x, self.ang_y, self.ang_z = [], [], []

    def _on_get_data(self, data: list):
        """
        传感器数据回调函数
        :param data: 数据
        :return:
        """
        for i in range(len(data) - 10):
            if data[i] == 0x55 and data[i + 1] == 0x51:
                axl, axh, ayl, ayh, azl, azh, *_ = data[i + 2:i + 11]
                self.acc_x.append((short((axh << 8) | axl)) / 32768 * 16 * 9.8)
                self.acc_y.append((short((ayh << 8) | ayl)) / 32768 * 16 * 9.8)
                self.acc_z.append((short((azh << 8) | azl)) / 32768 * 16 * 9.8)
            if data[i] == 0x55 and data[i + 1] == 0x52:
                wxl, wxh, wyl, wyh, wzl, wzh, *_ = data[i + 2:i + 11]
                # TODO: 这里每两个数就有一个是0，目前不知道为什么，影响效果先去除
                self.gyro_x.append((short(wxh << 8) | wxl) / 32768 * 2000)
                self.gyro_y.append((short(wyh << 8) | wyl) / 32768 * 2000)
                self.gyro_z.append((short(wzh << 8) | wzl) / 32768 * 2000)
                if len(self.gyro_x) > 2 and self.gyro_x[-1] and not self.gyro_x[-2] and self.gyro_x[-3]:
                    self.gyro_x[-2] = self.gyro_x[-1]
                    del self.gyro_x[-1]
                    self.gyro_y[-2] = self.gyro_y[-1]
                    del self.gyro_y[-1]
                    self.gyro_z[-2] = self.gyro_z[-1]
                    del self.gyro_z[-1]
            if data[i] == 0x55 and data[i + 1] == 0x53:
                rol, roh, pil, pih, yal, yah, *_ = data[i + 2:i + 11]
                self.ang_x.append((short(roh << 8 | rol) / 32768 * 180))
                self.ang_y.append((short(pih << 8 | pil) / 32768 * 180))
                self.ang_z.append((short(yah << 8 | yal) / 32768 * 180))

    @staticmethod
    def _get_sensor():
        """
        获取传感器
        :return: sensor
        """
        # 获取传感器hid
        hid_devices = hid.core.find_all_hid_devices()
        sensor = None
        for hid_device in hid_devices:
            if "Wit-Motion" in hid_device.product_name:
                sensor = hid_device
        if not sensor:
            raise Exception("没有设备")
        return sensor

    def set_handler(self):
        """
        注册回调函数，注册了之后才可以收到数据
        :return:
        """
        # 打开设备
        self.sensor.open()
        # 注册回调函数
        self.sensor.set_raw_data_handler(self._on_get_data)
