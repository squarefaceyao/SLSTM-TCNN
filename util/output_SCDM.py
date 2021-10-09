import time
import seaborn as sns
from pylab import mpl
import numpy as np
from keras.utils import np_utils
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
now = int(round(time.time()*1000))
localtime = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now/1000))

"""
功能：输出SCDM模型每一层的输出
输出保存在：plt.savefig(f'../figure/{model_name}_output2_{localtime}.jpg',dpi=400)
"""
sns.set_context("talk",font_scale=4)
sns.set_style("white")

# fontsize = 24
length = 20
width = 16
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

wheat = 'DK'
model_name = 'SCDM' # SCDM+SLSTM_TCNN

df_T = pd.read_csv(f'../Datasets/{wheat}_1.csv')

print(f'整个数据集的数量{df_T.shape[1]}')

#区分ab波和标签
all_ = df_T.shape[1]
ab = all_ - 1

xx = np.array(df_T.iloc[:,0:ab])
yy = np.array(df_T.iloc[:,ab:all_])
yy2 = np.squeeze(yy) # 去掉一个空维度

yy_ = np_utils.to_categorical(yy2,9) # 标签向量化
xx_ = xx.reshape(df_T.shape[0], ab, 1) # 波形扩充一个维度
if model_name == 'SCDM':
    model = keras.models.load_model('../model/DK_9fenlei_acc_0.45.h5')
else:
    model = keras.models.load_model('../model/DK_9fenlei_acc_0.74.h5')


model.summary()
print()
# model.summary()
layername=['nor1','Conv1','Max1','nor2','Conv2','Max2']
test=['BatchNormalization','Conv1D','MaxPooling','BatchNormalization','Conv1D','MaxPooling']
if model_name == 'SCDM':
    layername2=['flatten','Den1','dropout_4','Den2']
    test2=['Flatten','Full connection','Dropout','Output']
else:
    layername2=['flatten','Den1','dropout','Den2']
    test2=['Flatten','Full connection','Dropout','Output']


num = np.array(layername).shape[0]
num2 = np.array(layername2).shape[0]
#
plt.figure(figsize=(length, width))
plt.tight_layout()

# ##############################################################################################################
# 输出layername层的信息
for name,x in zip(layername,range(0, num)):
    # print(name)
    # print(x)
    sub_model = keras.models.Model( inputs = model.input, outputs = model.get_layer(name).output )
    l = sub_model.predict(xx_)
    print(l.shape)
    ax = plt.subplot(3, 2, x + 1)
    label_tmp = l[ :, :, x]
    plt.plot(label_tmp.reshape(l.shape[1],1))
    # plt.scatter(label_tmp.reshape(l.shape[1],1))
    plt.title(f'{test[x]}')
plt.tight_layout()
plt.savefig(f'../figure/{model_name}_output1_{localtime}.jpg',dpi=400)
plt.close()

###############################################################################################################
# 输出layername2 层的信息.输入的数据 必须是一个
plt.figure(figsize=(length, width))

for name,x in zip(layername2,range(0, num2)):
    print(name)
    print(x)
    sub_model = keras.models.Model( inputs = model.input, outputs = model.get_layer(name).output )
    l = sub_model.predict(xx_)
    print(l.shape)
    ax = plt.subplot(3, 2, x + 1)

    if name=='Den2':
        plt.plot(l.reshape(l.shape[1], 1), 'o')
        plt.title(f'{test2[x]}')
    else:
        plt.plot(l.reshape(l.shape[1], 1))
        plt.title(f'{test2[x]}')


plt.tight_layout()
plt.savefig(f'../figure/{model_name}_output2_{localtime}.jpg',dpi=400)
###############################################################################################################

# 输出output层的信息
# sub_model = keras.models.Model(inputs=model.input, outputs=model.get_layer('Den2').output)
# l = sub_model.predict(xx_)
# pd.DataFrame(l).to_csv('../result/SCDM_output.csv',index=False)