import seaborn as sns
from pylab import mpl
import numpy as np
from keras.utils import np_utils
import pandas as pd
from util.k_fold import kfold_cnn_9
sns.set(font_scale=2)
fontsize = 24
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

wheat = 'DK'
epochs = 1
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

kfold_cnn_9(inputs=xx_, targets=yy_,epochs=600,wheat=wheat)
