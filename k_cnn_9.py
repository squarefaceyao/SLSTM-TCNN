import seaborn as sns
from pylab import mpl
import numpy as np
from keras.utils import np_utils
import pandas as pd
from util.k_fold import kfold_cnn_9
import time

sns.set(font_scale=2)
fontsize = 24
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

wheat = 'DK'
expr_num = 30
df_T = pd.read_csv('Datasets/180条-{}-0_400mM-9fenlei.csv'.format(wheat))

print(f'整个数据集的数量{df_T.shape[0]}')

#区分ab波和标签
all_ = df_T.shape[1] # 147
ab = all_ - 1 # 146

xx = np.array(df_T.iloc[:,0:ab]) # df_T.iloc[:,0:146]
yy = np.array(df_T.iloc[:,ab:all_]) # df_T.iloc[:,146:147]
yy2 = np.squeeze(yy) # 去掉一个空维度

yy_ = np_utils.to_categorical(yy2,9) # 标签向量化
xx_ = xx.reshape(df_T.shape[0], ab, 1) # 波形扩充一个维度

i_Mean_acc, i_Mean_kappa, i_Mean_ham= [],[],[]
i_Std_acc, i_Std_kappa, i_Std_ham= [],[],[]
test={}
for i in range(expr_num):

    result = kfold_cnn_9(inputs=xx_, targets=yy_,epochs=600,wheat=wheat,save_model=1)

    test[f'{i}'] =result
#
# i_Mean_acc.append(result['acc_mean'])
# i_Mean_kappa.append(result['kappa_mean'])
# i_Mean_ham.append(result['ham_mean'])
# i_Std_acc.append(result['acc_std'])
# i_Std_kappa.append(result['kappa_std'])
# i_Std_ham.append(result['ham_std'])
now = int(round(time.time()*1000))
localtime = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now/1000))
pd.DataFrame(test).to_csv(f"result/SCDM_{wheat}_{localtime}.csv")