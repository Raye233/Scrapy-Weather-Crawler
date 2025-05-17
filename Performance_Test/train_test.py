import cProfile
import subprocess
import os

current_path = os.getcwd()
train_py_path = r'F:\Rayedata2\weather_crawl\weather_predict'

os.chdir(train_py_path)

command_1 = [
    'python',
    '-m',
    'cProfile',
    '-o',
    f'{current_path}/train_log.profile',
    '-s',
    'cumtime',
    'tensorflow_train.py'
]
subprocess.call(command_1)

os.chdir(current_path)

command_2 = [
    'snakeviz.exe',
    '-p',
    '8080',
    'train_log.profile'
]
subprocess.call(command_2)
