import numpy as np
import pandas as pd
from util.data_ultimately import data_ultimately,data_dispose
import os
import time
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns

now = int(round(time.time()*1000))
localtime = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now/1000))
sns.set_context("talk",font_scale=2.5)
sns.set_style("white")
"""
功能：输出SLSTM_TCNN模型每一层的输出
输出保存在：plt.savefig(f'../figure/{model_name}_output2_{localtime}.jpg',dpi=400)
"""

wheat = 'LD' # DK or LD
df_T = pd.read_csv('../Datasets/DK_DS2_1.csv').T

xx1,yy1 = data_dispose(df_T) # 调用切分数据函数
xx = np.array(xx1) # /100
yy = np.array(yy1) # /100
inputs = xx.reshape(len(xx),xx.shape[1],1) # 转换成三个维度
targets = yy.reshape(len(yy),yy.shape[1]) # 转换成二个维度
xx_ =  inputs
# 加载模型
units=12# 78、6、12 效果最好
# 预测-model_3效果最好
print(inputs.shape)
test_a = inputs
test_b = targets[:, :-1]
test_label = targets[: ,-1]*100
test_label = test_label.astype(np.int32)
if wheat == 'DK':
    model = keras.models.load_model('../model/DK_pcc_0.91.h5')
else:
    model = keras.models.load_model('../model/LD_pcc_0.89.h5') # load模型
model.summary()

model_name = 'SLSTM_TCNN'
layername=['nor1','Max1','Conv1','tf_op_layer_mul_8','lstm_1','TC1']
if model_name == 'SLSTM_TCNN':
    layername2=['flatten','dense_25','dropout_8','dense_26']
else:
    layername2=['flatten','dense_25','dropout_8','dense_26']


num = np.array(layername).shape[0]
num2 = np.array(layername2).shape[0]
#
length = 22
width = 16
plt.figure(figsize=(length, width))
plt.tight_layout()

# ##############################################################################################################
# 输出layername层的信息
test2 = ['BatchNormalization','MaxPooling','Conv1D','Concat','LSTM','UpSampling']
for name,x in zip(layername,range(0, num)):
    print(name)
    print(x)
    sub_model = keras.models.Model( inputs = model.input, outputs = model.get_layer(name).output )
    l = sub_model.predict(xx_)
    # print(l.shape)
    ax = plt.subplot(3, 2, x + 1)
    # label_tmp = l[:,:, x+1]
    label_tmp = l[:, :,-1]
    # print(label_tmp.shape)
    plt.plot(label_tmp.reshape(l.shape[1],1))
    # plt.scatter(label_tmp.reshape(l.shape[1],1))
    plt.title(f'{test2[x]}')
plt.tight_layout()
plt.savefig(f'../figure/{wheat}_{model_name}_output1_{localtime}.jpg',dpi=400)
plt.close()

##############################################################################################################
# 输出layername2 层的信息.输入的数据 必须是一个
plt.figure(figsize=(length, width))
test=['Flatten','Full connection','Dropout','Output']
for name,x in zip(layername2,range(0, num2)):
    print(name)
    print(x)
    sub_model = keras.models.Model( inputs = model.input, outputs = model.get_layer(name).output )
    l = sub_model.predict(xx_)
    print(l.shape)
    ax = plt.subplot(3, 2, x + 1)

    if name=='Den2':
        plt.plot(l.reshape(l.shape[1], 1), 'o')
        plt.title('SoftMax')
    else:
        plt.plot(l.reshape(l.shape[1], 1))
        plt.title(f'{test[x]}')


plt.tight_layout()
plt.savefig(f'../figure/{wheat}_{model_name}_output2_{localtime}.jpg',dpi=400)