import numpy as np
import pandas as pd
import math
import seaborn as sns
from sklearn.model_selection import KFold
from util.data_ultimately import data_ultimately,data_dispose
from util.model_1 import model_4
from util.predict_assess import assessment
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
localtime = time.asctime( time.localtime(time.time()) )
import datetime
starttime = datetime.datetime.now()


'''
 训练全部的数据，测试集根据标签选择需要预测的浓度
1. 读取全部的数据
2. 切分数据 ，标签在b波后面
3. 用下面的代码切分数据。但是target里面是有标签的，
    kfold = KFold(n_splits=num_folds, shuffle=True)
    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(inputs, targets):
4. 把targets里面的标签提取出来
5. 在训练模型
6. 根据标签打印全部的评价指标
'''
now = int(round(time.time()*1000))
localtime = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(now/1000))
# plt 图片大小
sns.set(font_scale=1.5)
fontsize = 21
# 保存评价指标

#输入的参数
expriment_num = 1  # 试验次数。进行几次交叉验证
epochs = 600 # 训练次数，600
n_splits = 10 # 折数，n_splits折交叉验证,10
wheat = 'DK' # DK or LD

# 保存训练的模型
model_savePath = 'model/{}不同盐浓度预测模型/{}/'.format(wheat,localtime)
if os.path.exists(model_savePath) == True:
    print('每个盐浓度预测时候的模型保存在文件夹:"{}"'.format(model_savePath))
else:
    os.makedirs(model_savePath)
    print('每个盐浓度预测时候的模型保存在文件夹:"{}"'.format(model_savePath)) 

# df_T = pd.read_csv('../数据集/{}-9预测数据集标签是0-8.csv'.format(wheat)).T

if wheat == 'DK':
    df_T = pd.read_csv(f'Datasets/{wheat}_DS1_240.csv').T
else:
    df_T = pd.read_csv('Datasets/LD_DS1_163.csv').T

xx1,yy1 = data_dispose(df_T) # 调用切分数据函数
xx = np.array(xx1, dtype=np.float32) # /100
yy = np.array(yy1, dtype=np.float32) # /100

inputs = xx.reshape(len(xx),xx.shape[1],1) # 转换成三个维度
targets = yy.reshape(len(yy),yy.shape[1]) # 转换成二个维度


# K 折交叉验证
num_folds = int(math.log(xx.shape[1]))+1

kfold = KFold(n_splits=n_splits, shuffle=True)
# K-fold Cross Validation model evaluation
i_Mean_MSE, i_Mean_pcc, i_Mean_fre= [],[],[]
i_Std_MSE, i_Std_pcc, i_Std_fre= [],[],[]
CV_Mean_MSE, CV_Std_MSE = [],[]
CV_Mean_pcc, CV_Std_pcc = [],[]
CV_Mean_fre, CV_Std_fre = [],[]
#Repeat experiment i times
for i in range(expriment_num):
    print(f'这是第{i}次实验')
    pcc_per_fold, fre_per_fold,mse_per_fold = [], [], []
 
    fold_no = 1
    for train, test in kfold.split(inputs, targets):
        # 加载模型
        units=12# 78、6、12 效果最好
        # 预测-model_3效果最好
        print(inputs[train].shape)
        print(inputs[test].shape)
        train_a = inputs[train]
        test_a = inputs[test]

        train_b = targets[train][:, :-1]
        test_b = targets[test][:, :-1]

        train_label = targets[train][: ,-1]*100
        test_label = targets[test][: ,-1]*100

        units=12# 78、6、12 效果最好
        batch_size= 64 # 22 效果最好，调成64.效果也行。
        model = model_4(units=units,
                        input_size=train_a.shape[1]) # 预测-model_3效果最好

        model.compile(loss='mean_squared_error',
                    optimizer='Adam', 
                    metrics=['mae'])
        # 模型训练
        # verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
        history = model.fit(train_a,train_b, 
                            epochs=epochs,
                            verbose=0,
                            validation_split=0.1) # ephos=1000 容易过拟合

        l = model.predict(test_a)
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...') 

        # 总体评价指标             
        ave_pcc,ave_fre,ave_mse = assessment(test_b=test_b,
                                                l=l,
                                                salt='ALL_In',wheat=wheat)
        # 保存每一折里面的评价指标。
        pcc_per_fold.append(ave_pcc)
        fre_per_fold.append(ave_fre)
        mse_per_fold.append(ave_mse)
        # ave_mae = np.mean(history.history['val_mae'])
        model.save(model_savePath + 'pcc_{}.h5'.format(ave_pcc))
        # model = keras.models.load_model(model_savePath + '{}_{}浓度mae_{}.h5'.format(wheat,test[salt],ave_mae))
        # Increase fold number
        fold_no = fold_no + 1

    # 这里保存每次实验的结果 10折交叉验证评价指标的平均值
    i_Mean_MSE.append(np.mean(mse_per_fold))
    i_Mean_pcc.append(np.mean(pcc_per_fold))
    i_Mean_fre.append(np.mean(fre_per_fold))

    # 这里保存每次实验的结果 10折交叉验证评价指标的方差
    i_Std_MSE.append(np.std(mse_per_fold))
    i_Std_pcc.append(np.std(pcc_per_fold))
    i_Std_fre.append(np.std(fre_per_fold))

    # 这里输出每次实验的结果
    print('------------------------------------------------------------------------')
    print('pcc fre mae per fold')
    for i in range(0, len(pcc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i+1} - pcc: {pcc_per_fold[i]} - fre: {fre_per_fold[i]} - mse:{mse_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> pcc: {np.mean(pcc_per_fold)} (+- {np.std(pcc_per_fold)})')
    print(f'> fre: {np.mean(fre_per_fold)} (+- {np.std(fre_per_fold)})')
    print(f'> mse: {np.mean(mse_per_fold)} (+- {np.std(mse_per_fold)})')

    print('------------------------------------------------------------------------')


# 进行格式转换，保存为csv格式
#Convert to numpy for convenience
CV_Mean_MSE  = np.asarray(i_Mean_MSE)
CV_Std_MSE  = np.asarray(i_Std_MSE)
#Convert to numpy for convenience
CV_Mean_pcc = np.asarray(i_Mean_pcc)
CV_Std_pcc = np.asarray(i_Std_pcc)

CV_Mean_fre = np.asarray(i_Mean_fre)
CV_Std_fre = np.asarray(i_Std_fre)

df = pd.DataFrame({"CV_Mean_MSE" : CV_Mean_MSE,
                   "CV_Std_MSE" : CV_Std_MSE,
                   "CV_Mean_pcc" : CV_Mean_pcc,
                   "CV_Std_pcc" : CV_Std_pcc,
                   "CV_Mean_fre" : CV_Mean_fre,
                   "CV_Std_fre" : CV_Std_fre
                   })

df.to_csv(f"./result/SLSTM_TCNN_{wheat}_{localtime}_CV_Result.csv", index=False)
    
        


