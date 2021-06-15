import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from pylab import mpl
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from numpy import argmax

sns.set(font_scale=2)
fontsize = 24
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

def cover(test_y):
    test = []
    for i in range(test_y.shape[0]):
        test.append(argmax(test_y[i]))
    yy = np.array(test)
    yy2 = np.squeeze(yy)
    return yy2

def eva_2_class(l,test_y,datasets,figure_savePath):
    s = 0
    for i in range(len(l)):
        if np.argmax(l[i]) == np.argmax(test_y[i]):
            s+=1
    # 计算acc
    acc = round(s/len(l),4)
    y_true = cover(test_y)
    y_pred = cover(l)
    y_score = l[:,1]
    # 计算AUC https://www.tensorflow.org/api_docs/python/tf/keras/metrics/AUC
    m = tf.keras.metrics.AUC(num_thresholds=3)
    m.update_state(y_pred, y_score)
    auc = m.result().numpy()

    # 输出评价报告
    # target_names = ["0mm",'50mm','100mm', '150mm','200mm', '250mm','300mm','350mm','400mm']
    # print(classification_report(y_true, y_pred, target_names=target_names))
    # 绘制混淆矩阵
    cf = confusion_matrix(y_true, y_pred)
    x_tick = ['DK','LD']
    y_tick = ['DK','LD']
    pd_data=pd.DataFrame(cf,index=y_tick,columns=x_tick)
    total1 = sum(sum(cf))

    sensitivity1 = cf[0, 0] / (cf[0, 0] + cf[0, 1])
    print('Sensitivity : ', sensitivity1)
    f, ax = plt.subplots(figsize=(5, 4))
    cmap = sns.cm.rocket_r
    ax = sns.heatmap(pd_data, annot=True, ax=ax, linewidths=0.5,fmt='d', cmap='BuPu') #画heatmap，具体参数可以查文档
    plt.rcParams.update({"font.size":fontsize})
    plt.xlabel('Predicted Label',fontsize=fontsize, color='k') #x轴label的文本和字体大小
    plt.ylabel('True Label',fontsize=fontsize, color='k') #y轴label的文本和字体大小
    plt.xticks(rotation=40,fontsize=fontsize) #x轴刻度的字体大小（文本包含在pd_data中了）
    plt.yticks(fontsize=fontsize) #y轴刻度的字体大小（文本包含在pd_data中了）
    plt.title(f'{datasets}-{l.shape[0]}-2-clss',fontsize=fontsize) #图片标题文本和字体大小

    figername = figure_savePath+f"{datasets}_acc is{acc}_number is {l.shape[0]}_2_clss.png"
    plt.savefig(figername,dpi=400)
    plt.close()
    print(f"已保存2分类混淆矩阵,保存路径是{figername}")

    # 计算卡帕系数和海明距离

    acc = np.round(acc,4)

    auc = np.round(auc,4)

    sen = np.round(sensitivity1,4)
    return acc,auc,sen