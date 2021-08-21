r'''
Author       : PiKaChu_wcg
Date         : 2021-08-22 04:39:06
LastEditors  : PiKachu_wcg
LastEditTime : 2021-08-22 06:27:42
FilePath     : \flask-video-streaming-recorder\controller\modules\home\__init__.py
'''
r'''
Author       : PiKaChu_wcg
Date         : 2021-08-22 04:39:06
LastEditors  : PiKachu_wcg
LastEditTime : 2021-08-22 04:59:55
FilePath     : \flask-video-streaming-recorder\controller\modules\home\__init__.py
'''
from flask import Blueprint
from efficientdet_test import detecter
# 创建蓝图对象
home_blu = Blueprint("home", __name__)
global_var=detecter("weights/efficientdet-d0.pth")
# 让视图函数和主程序建立关联
from controller.modules.home.views import *
