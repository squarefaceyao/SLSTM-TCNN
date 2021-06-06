
"""
制作DSC1-9数据集。

输入是：LD_DSC_180.csv 和 DK_DSC_180.csv.
输出是：DSC_1_20.csv DSC_2_20.csv DSC_3_20.csv DSC_4_20.csv DSC_5_20.csv DSC_6_20.csv DSC_7_20.csv DSC_8_20.csv  DSC_9_20.csv
标签（DK--0,LD--1）
"""
import numpy as np
from util.data_ultimately import data_ultimately, data_dispose
import pandas as pd
import os
df_DK = pd.read_csv(f'../Datasets/DK_DSC_180.csv')
df_LD= pd.read_csv(f'../Datasets/LD_DSC_180.csv')

def split_a_b_label(df_DK,wheat):
    '''

    :param df_DK: 输入LD_DSC_180.csv
    :return: 输出ab波 label 标签（DK--0,LD--1）
    '''

    aa_DK,bb_DK = data_dispose(df_DK)
    print(f'b波的长度是{bb_DK.shape[1]}')
    label_DK = bb_DK[:, -1] * 100.0
    aa_DK = np.squeeze((aa_DK)) # 去掉a波的空余维度
    ab_DK = np.hstack((aa_DK, bb_DK))  # 合并a和b波
    ab_DK = ab_DK[:, :-1]

    # 这里做的是区分品种，所以需要生成新的标签
    if wheat == 'DK':
        y = np.array([1] * ab_DK.shape[0])  # DK 生成标签

    else:
        y = np.array([0] * ab_DK.shape[0])  # LD 生成标签

    y = y[:, np.newaxis]  # y 扩充一个维度，拼接到x的后面 z

    data = np.hstack((ab_DK, y))

    return data,label_DK.astype(int)

ab_DK,label_DK = split_a_b_label(df_DK,'DK')
ab_LD,label_LD = split_a_b_label(df_LD,'LD')

savePath = '../Datasets/DSC/'
if os.path.exists(savePath) == True:
    print(f'每个盐浓度的文件保存在{savePath}')
else:
    os.makedirs(savePath)
    print(f'每个盐浓度的文件保存在{savePath}')


# salt = 0
for salt in range(9):
    _ab_DK = ab_DK[label_DK == salt, :]  # 在train上根据标签选择ab
    _ab_LD = ab_LD[label_LD == salt, :]  # 在train上根据标签选择ab

    x = np.vstack((_ab_DK, _ab_LD))

    pd.DataFrame(x).to_csv(savePath+f'DSC{salt+1}_{x.shape[0]}.csv',index=False)

    print(x.shape)