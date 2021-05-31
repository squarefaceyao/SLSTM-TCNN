import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from util.data_ultimately import data_ultimately
import math
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from util.frechet import get_similarity
import seaborn as sns
import os
import time
localtime = time.asctime( time.localtime(time.time()) )
# plt 图片大小
sns.set(font_scale=1.5)
fontsize = 32

def show_wave(l,test_a,test_b,salt,wheat,path):
    predict_b_savePath = path
    test = ['0mM','50mM','100mM','150mM','200mM','250mM','300mM','350mM','400mM']
    plt.figure(figsize=(12,9))
    num = l.shape[0]    
    leng = l.shape[1]
    x1 = np.linspace(0,leng,leng) #numpy.linspace(开始，终值(含终值))，个数)
    x2 = np.linspace(leng,leng+leng,leng)
    for n in range(num):
        # print('正在保存第 {} 预测的波形'.format(n))
        y_pred = l[n]
        x_true = np.squeeze(test_a[n])
        y_true = test_b[n]
        if n==0:
            plt.plot(x1, x_true, color='black', label='No Stimulation Values')
            plt.plot(x2, y_true, 'b', label='True Stimulation Values')#'b'指：color='blue'
            plt.plot(x2, y_pred, 'g', label='Predictions')#'b'指：color='blue'
            plt.legend()  #显示上面的label
            
        else:
            plt.plot(x1, x_true, color='black')
            plt.plot(x2, y_true, 'b')#'b'指：color='blue'
            plt.plot(x2, y_pred, 'g')#'b'指：color='blue'
            plt.legend()  #显示上面的label
            
        plt.axis([0, leng+leng,-1.25,1.5])#设置坐标范围axis([xmin,xmax,ymin,ymax])
        plt.xlabel('Time(s)',fontsize=fontsize, color='k') #x轴label的文本和字体大小
        plt.ylabel('Normalized amplitude',fontsize=fontsize, color='k') #y轴label的文本和字体大小
        plt.xticks(rotation=0,fontsize=fontsize) #x轴刻度的字体大小（文本包含在pd_data中了）
        plt.yticks(fontsize=fontsize) #y轴刻度的字体大小（文本包含在pd_data中了）
        plt.title('{} {} salt stimulation'.format(wheat,test[salt]),fontsize=fontsize)  #标题
        plt.grid(linestyle='-.')
    # plt.show()
    plt.savefig( predict_b_savePath +'{} {} salt stimulation-{}'.format(wheat,test[salt],num),dpi=400)
    plt.close()

    
def assessment(test_b,l,salt,wheat):
    pp = '{}在{}salt测试集上的数量={}'.format(wheat,salt,l.shape[0])
    print('\033[1;37;43m   {}!\033[0m'.format(pp))
    # 测试集的皮尔逊系数   
    test_y = test_b 
    pcc = []
    for i in range(0,test_y.shape[0]):
        pccs = pearsonr(test_y[i],l[i])
        pcc.append(pccs[0])
    ave_pcc = np.mean(pcc)
    # 测试集的frechet
    frechet = []
    for i in range(0,test_y.shape[0]):
        fr = get_similarity(test_y[i],l[i])
        frechet.append(fr)
    ave_fre = np.mean(frechet)
    # 测试集的rmse
    rmse = []
    for i in range(0,test_y.shape[0]):
        rm = np.sqrt(mean_squared_error(test_y[i],l[i]))
        rmse.append(rm)
    ave_mse = np.mean(rmse)
    print("\033[32m!!! Good Luck !!!  \033[0m") # 调整terminal 输出的颜色。
    return(ave_pcc,ave_fre,ave_mse)