r'''
Author       : PiKaChu_wcg
Date         : 2021-08-22 04:39:06
LastEditors  : PiKachu_wcg
LastEditTime : 2021-08-22 07:40:01
FilePath     : \flask-video-streaming-recorder\main.py
'''
# from flask_script import Manager
from controller import create_app
from controller.modules.home import global_var

# 创建APP对象
app = create_app('pro')
# # 创建脚本管理
# mgr = Manager(app)


if __name__ == '__main__':
    # mgr.run()
    app.run(
        # threaded=True,
        host='0.0.0.0',
        port=5000
        )

