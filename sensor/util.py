"""
工具支持
"""
import os


def get_static_file_full_path(file_name: str):
    """
    返回静态文件的绝对路径，用于加载图片时候的路径
    :param file_name:
    :return:
    """

    return os.path.join(os.path.dirname(__file__), "static", file_name)


if __name__ == "__main__":
    print(get_static_file_full_path("1.png"))
