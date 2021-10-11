import time

from numpy import argmax
from fnmatch import fnmatch
from sklearn.model_selection import train_test_split

from util.K_STCM_SLSTM_TCMM import kfold_STCM_SLSTM_TCNN
from util.k_fold import kfold_cnn_2
from util.model_1 import cnn_model_2
from util.eva_2_class import eva_2_class

import numpy as np
from keras.utils import np_utils
import pandas as pd
from util.data_ultimately import data_ultimately,data_dispose
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import datetime

def cover(test_y):
    test = []
    for i in range(test_y.shape[0]):
        test.append(argmax(test_y[i]))
    yy = np.array(test)
    yy2 = np.squeeze(yy)
    return yy2

'''
根据标签预测数据。
并把每个类里的测试集的a波保存起来
'''

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

### main()
wheat1 = 'DK'  # 目的是为了生成标签 0
wheat2 = 'LD'
salt_value = ['0mM','50mM','100mM','150mM','200mM','250mM','300mM','350mM','400mM']


savePath = 'Datasets/DSCP/'

# range(9)

'''
1 读取数据 一个文件夹里的全部数据集。根据字符串名字规律定位使用哪个模型

'''
salt = ['0mM','50mM','100mM','150mM','200mM','250mM','300mM','350mM','400mM']
epochs = 600
result={}
for i in range(9):
    name = [name for name in os.listdir(savePath)
                  if fnmatch(name, f'DSCP{i+1}_148.csv')]  # 获取path路径后缀为.h5的文件
    pa = str(name)  # 转换成str类型，通过 pa[2:-2]获取文件名字
    all_data = pd.read_csv(savePath+pa[2:-2]) # load模型

    ab = all_data.shape[1]
    xx = all_data.iloc[:, 0:ab-1]
    xx = np.array(xx)
    xx_ = xx.reshape(len(xx), ab - 1, 1)  # 扩冲一个维度

    yy = all_data.iloc[:, ab-1:ab]
    yy = np.array(yy) # array转换
    yy2 = np.squeeze(yy) # 去掉一个维度
    y_train = np_utils.to_categorical(yy2, 2)  # 标签向量化

    print(f'输入数据的ab波shape：{xx_.shape}, 标签shape是: {y_train.shape}')

    # 调用交叉验证函数
    print(f'训练次数为: {epochs}')
    result[salt[i]] = kfold_STCM_SLSTM_TCNN(inputs=xx_, targets=y_train,salt=i,
                                  datasets = pa[2:-2],epochs=epochs,
                                  save_model=1)

now = int(round(time.time() * 1000))
localtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now / 1000))
pd.DataFrame(result).to_csv(f"result/STCM_SLSTM_TCNN_{localtime}.csv")
