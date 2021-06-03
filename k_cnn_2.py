
from numpy import argmax
from util.k_fold import kfold,kfold_cnn_2
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

# 区分x和标签
def test(df_T,salt,wheat,DS):
    """
    df_T: 输入文件的格式是{}-9预测数据集标签是0-8.csv
    wheat：小麦品种 DK or LD
    salt: 盐浓度标签 0、1、2、3、4、5、6、7、8代表'0mM','50mM','100mM','150mM','200mM','250mM','300mM','350mM','400mM'
    功能：根据标签选择返回ab波，标签（DK--0,LD--1）
    
    """
    # data_ultimately（）返回的是ab波
    if DS == 'CNN+SLSTM-TCNN':
        num = 1177
    else:
        num = 1177
    salt_value = ['0mM', '50mM', '100mM', '150mM', '200mM', '250mM', '300mM', '350mM', '400mM']

    print(f'接收到的数据个数:{df_T.shape[1]}和盐刺激浓度:{salt_value[salt]}，正在处理的品种:{wheat}')

    train_a,test_a,train_b,test_b,train_label,test_label = data_ultimately(df_T,num=num,test_size=0.2,random_state=9) # num 提取原始数据的特征值，1176 效果也行 ,random_state=20        147
    train_a = np.squeeze((train_a))# 去掉a波的空余维度
    test_a = np.squeeze((test_a)) # 去掉a波的空余维度
    train_ab = np.hstack((train_a,train_b)) # 合并a和b波
    test_ab = np.hstack((test_a,test_b))# 合并a和b波
    train_x = train_ab[train_label == salt,:]# 在train上根据标签选择ab
    test_x = test_ab[test_label == salt,:] # 在test上根据标签选择ab

    x = np.vstack((train_x,test_x))
    if wheat == 'DK':
        y = np.array([1] * x.shape[0])# DK 生成标签
        
    else:
        y = np.array([0] * x.shape[0])# LD 生成标签
    
    y = y[:,np.newaxis] # y 扩充一个维度，拼接到x的后面
    data = np.hstack((x,y))
    if DS =='CNN':
        data2 = data[0:20]
    else:
        data2 = data
    #任务 保证每次输出的长度为20
    # print("数据的长度为{}".format(data2.shape))
    imput_size = x.shape[1]
    return data2,imput_size

### main()
wheat1 = 'DK'  # 目的是为了生成标签 0
wheat2 = 'LD' 
epochs = 300
salt_value = ['0mM','50mM','100mM','150mM','200mM','250mM','300mM','350mM','400mM']

DS = 'CNN' # or CNN+SLSTM-TCNN

if DS == 'CNN':
    df_T_DK = pd.read_csv('Datasets/180条-{}-0_400mM-9fenlei.csv'.format(wheat1)).T # 真实数据中300mM的数据比较多，需要切片，保证每个类别下都是20条数据。
    df_T_LD = pd.read_csv('Datasets/180条-{}-0_400mM-9fenlei.csv'.format(wheat2)).T
if DS == 'CNN+SLSTM-TCNN':   
    df_T_DK = pd.read_csv('../数据集/702条-{}-0_400mM-9fenlei.csv'.format(wheat1))
    df_T_LD = pd.read_csv('../数据集/567条-{}-0_400mM-9fenlei.csv'.format(wheat2)) # 扩充的数据集，不需要转置
# range(9)
for salt in range(3):
    print("这是盐浓度{}下的数据".format(salt_value[salt]))

    data_DK,imput_size_DK = test(df_T_DK,salt,wheat1,DS)
    data_LD,imput_size_LD = test(df_T_LD,salt,wheat2,DS)
    all_data = np.vstack((data_DK,data_LD)) # 合并两个品种的数据
    # print("读取所有 的数据：",all_data.shape)
"""
    pd.DataFrame(all_data).to_csv("Datasets/盐浓度{}下的数据.csv".format(salt_value[salt]))
    # print(all_data.shape)
    # 切分数据
    
    ab = all_data.shape[1]
    xx = all_data[:,0:ab-1] # ab波
    yy =all_data[:,ab-1:ab] # DK和LD的标签
    xx = np.array(xx)/100 # 归一化 /100
    yy = np.array(yy) # array转换
    yy2 = np.squeeze(yy) # 去掉一个维度
    # print(yy2)
    y_train = np_utils.to_categorical(yy2,2) # 标签向量化
    xx_ = xx.reshape(len(xx),ab-1,1) # 扩冲一个维度

    inputs = xx_
    targets = y_train


    # k折交叉验证
    kfold_cnn_2(inputs, targets,salt,ab,epochs,model_name=DS)



"""

