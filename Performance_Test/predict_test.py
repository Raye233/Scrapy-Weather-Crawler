import cProfile
import subprocess
import os

current_path = os.getcwd()
predict_py_path = r'F:\Rayedata2\weather_crawl'

os.chdir(predict_py_path)

command_1 = [
    'python',
    '-m',
    'cProfile',
    '-o',
    f'{current_path}/predict_log.profile',
    '-s',
    'cumtime',
    'tensorflow_predict.py'
]
subprocess.call(command_1)

os.chdir(current_path)

command_2 = [
    'snakeviz.exe',
    '-p',
    '8080',
    'predict_log.profile'
]
subprocess.call(command_2)
