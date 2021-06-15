import tensorflow as tf
import numpy as np
import pandas as pd
from util.data_ultimately import data_ultimately,data_dispose
from util.predict_assess import show_wave,assessment
import os
import time
import datetime
from tensorflow import keras
now = int(round(time.time()*1000))
localtime = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now/1000))
import matplotlib.pyplot as plt

# 注意： 预测用的x需要不一样。
"""
使用预测模型
使用一个文件夹里的model，预测不同盐刺激下的波形。
使用的a波是训练模型时候的测试集。这里需要手工把salt_predict.py 里的testa_savePath文件手工合并到一起
并把a波和预测的b波拼接起来
"""
# 根据小麦品种选择模型保存的路径

path = "../model/LD-use" #文件夹目录
from fnmatch import fnmatch
canshu = [1.12,1.23,1.31,1.37,1.43,1.51,1.6,1.7,1.8] # 为了每次保存的a波形都不一样。波形乘的数字越大，幅值越低。
salt = [0,50,100,150,200,250,300,350,400] # 根据salt值选择模型和保存预测的数据
#输入的参数
wheat = 'DK' # DK or LD
test = ['0mM','50mM','100mM','150mM','200mM','250mM','300mM','350mM','400mM']
predict_b_savePath = './figer/{}不同盐浓度预测的b波/{}/'.format(wheat,localtime)
if os.path.exists(predict_b_savePath) == True:
    print('每个盐浓度预测时候的模型保存在文件夹:"{}"'.format(predict_b_savePath))
else:
    os.makedirs(predict_b_savePath)
    print('每个盐浓度预测时候的模型保存在文件夹:"{}"'.format(predict_b_savePath))

# df_T = pd.read_csv('../数据集/{}-9预测数据集标签是0-8.csv'.format(wheat)).T
df_T = pd.read_csv('../Datasets/DK_DS2_1.csv').T


xx1,yy1 = data_dispose(df_T) # 调用切分数据函数
xx = np.array(xx1) # /100
yy = np.array(yy1) # /100
inputs = xx.reshape(len(xx),xx.shape[1],1) # 转换成三个维度
targets = yy.reshape(len(yy),yy.shape[1]) # 转换成二个维度

# 加载模型
units=12# 78、6、12 效果最好
# 预测-model_3效果最好
print(inputs.shape)
test_a = inputs
test_b = targets[:, :-1]
test_label = targets[: ,-1]*100
test_label = test_label.astype(np.int32)
if wheat == 'DK':
    model = keras.models.load_model('../model/DK_pcc_0.9114367818861696.h5')
else:
    model = keras.models.load_model('../model/LD_pcc_0.9042618886525235.h5') # load模型
# model.summary()

sub_model = tf.keras.models.Model( inputs = model.input, outputs = model.get_layer('lstm_1').output )
# 查看一下子模型的结果：
sub_model.summary()

l = sub_model.predict(test_a)
print(l[:,:,1].shape)
# label_tmp = l[ :, x]

plt.figure(figsize=(24, 24))
for x in range(0, 12):
    ax = plt.subplot(3, 4, x + 1)
    label_tmp = l[ :, :, x]
    plt.plot(label_tmp.reshape(147,1))

    # 去除坐标轴
    plt.xticks([])
    plt.yticks([])
    plt.title(f'{x}')
    # 去除黑框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

plt.tight_layout()
plt.legend()
# plt.show()
plt.savefig('../figure/LSTM_submodel22.jpg',dpi=400)

