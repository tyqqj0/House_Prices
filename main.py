# -*- CODING: UTF-8 -*-
# @time 2023/2/13 21:51
# @Author tyqqj
# @File main.py

import numpy as np
import torch
from torch.utils import data
from torch import nn

import matplotlib.pyplot as plt
import pandas as pd

import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using ', device)

if __name__ == '__main__':
    # print(sys.path)
    # 这里之前读取遇到了问题，找不到路径
    # 之后发现是因为我把data文件夹放在了项目的venv根目录下，而不是和main.py同级
    #
    # 使用sys.path打印后发现问题，这里，还可以使用sys.path.append()来添加路径
    df_train = pd.read_csv('data/train.csv')
    print(df_train.head())
    print(df_train.columns)
