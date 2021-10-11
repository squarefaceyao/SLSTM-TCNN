# 使用LD_DS2(54条), DK_DS2(54条)的a波形预测9个盐浓度刺激下的b波形
"""
1 选择pcc最大的模型
2 根据标签读取不同盐刺激浓度的a波

"""
import numpy as np
import pandas as pd
from util.data_ultimately import data_ultimately,data_dispose
from util.predict_assess import show_wave,assessment
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
localtime = time.asctime( time.localtime(time.time()) )
import datetime
starttime = datetime.datetime.now()
from tensorflow import keras
now = int(round(time.time()*1000))
localtime = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now/1000))

# 注意： 预测用的x需要不一样。
"""
使用预测模型
使用一个文件夹里的model，预测不同盐刺激下的波形。
使用的a波是训练模型时候的测试集。这里需要手工把salt_predict.py 里的testa_savePath文件手工合并到一起
并把a波和预测的b波拼接起来
"""
# 根据小麦品种选择模型保存的路径

canshu = [1.12,1.23,1.31,1.37,1.43,1.51,1.6,1.7,1.8] # 为了每次保存的a波形都不一样。波形乘的数字越大，幅值越低。
salt = [0,50,100,150,200,250,300,350,400] # 根据salt值选择模型和保存预测的数据
#输入的参数
wheat = 'DK' # DK or LD
test = ['0mM','50mM','100mM','150mM','200mM','250mM','300mM','350mM','400mM']
for s in range(9):
    cc = canshu[s]
    df_T = pd.read_csv(f'Datasets/{wheat}_DS2_54_test.csv').T

    xx1,yy1 = data_dispose(df_T) # 调用切分数据函数
    xx = np.array(xx1)
    yy = np.array(yy1)
    inputs = xx.reshape(len(xx),xx.shape[1],1) # 转换成三个维度
    targets = yy.reshape(len(yy),yy.shape[1]) # 转换成二个维度


    print(f"输入的数据纬度{xx.shape}")
    test_a = inputs
    test_b = targets[:, :-1]
    test_label = targets[: ,-1]*100
    test_label = test_label.astype(np.int32)

    if wheat == 'DK':
        model = keras.models.load_model('model/DK_pcc_0.9114367818861696.h5')
    else:
        model = keras.models.load_model('model/pcc_0.9245995283126831.h5') # load模型

    l = model.predict(test_a)

    xx  = np.hstack((np.squeeze(xx), l))
    xx = np.round(xx,3)*100
    # 这里做的是区分品种，所以需要生成新的标签

    y = np.array([s] * xx.shape[0])  # DK 生成标签
    y = y[:, np.newaxis]  # y 扩充一个维度，拼接到x的后面 z
    all_data=np.hstack((xx,y))

    pd.DataFrame(all_data).to_csv('result/feak_{}_{}salt_predict_ab.csv'.format(wheat,salt[s]))  # 横向合并合并预测的b波

    print(f"输出的数据纬度{l.shape}")

    del model


