import math
import os
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from util.eva_2_class import eva_2_class
from util.frechet import get_similarity
import time
localtime = time.asctime( time.localtime(time.time()) )
from numpy import argmax
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from util.model_1 import model_3,cnn_model_2,cnn_model
from sklearn.metrics import roc_curve, auc
from scipy import interp
from util.eva_9_class import eva_9_class
import seaborn as sns


def k_SCDM_SLSTM_TCNN(inputs, targets,epochs,wheat,save_model):

    num_folds = 5
    acc_list, kappa_list, ham_distance_list,auc_list = [],[],[],[]
    mean_fpr = np.linspace(0, 1, 100)

    epochs = epochs
    salt_value = ['0mM', '50mM', '100mM', '150mM', '200mM', '250mM', '300mM', '350mM', '400mM']
    kfold = KFold(n_splits=num_folds, shuffle=True)
    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(inputs, targets):
        print(f'test datasets number is {inputs[test].shape[0]}')

        # 预测-model_3效果最好
        model = cnn_model(input_size=inputs[train].shape[1])

        # verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
        history = model.fit(inputs[train],
                            targets[train],
                            epochs=epochs,
                            verbose=0,
                            validation_split=0.2)  # ephos=1000 容易过拟合

        #   打印模型训练时的评价指标
        # Generate generalization metrics
        y_pred = model.predict(inputs[test])
        acc,kappa,ham_distance,auc = eva_9_class(l=y_pred, test_y=targets[test], wheat=wheat)
        # 保存每一折里面的评价指标
        acc_list.append(acc)
        kappa_list.append(kappa)
        ham_distance_list.append(ham_distance)
        auc_list.append(auc)

        if save_model == True:
            model_savePath = os.getcwd()+'/model/SCDM_SLSTM_TCNN_{}_/{}/'.format(wheat, localtime)
            if os.path.exists(model_savePath) == True:
                print('SCDM_SLSTM_TCNN save path:"{}"'.format(model_savePath))
            else:
                os.makedirs(model_savePath)
                print('SCDM_SLSTM_TCNN save path::"{}"'.format(model_savePath))
            model.save(model_savePath + f'{fold_no}kflod_acc_{acc}.h5')
            print('已经保存模型')
        else:
            print('没有保存模型')
        print(f'这是第{fold_no}折')
        fold_no = fold_no + 1

    print(f'预测的小麦品种为{wheat}')
    print('------------------------------------------------------------------------')
    print('pcc fre mae per fold')
    for i in range(0, len(acc_list)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - acc: {acc_list[i]} - kappa: {kappa_list[i]} - ham_distance:{ham_distance_list[i]}- auc_list:{auc_list[i]}')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> acc: {np.mean(acc_list)} (+- {np.std(acc_list)})')
    print(f'> kappa: {np.mean(kappa_list)} (+- {np.std(kappa_list)})')
    print(f'> ham_distance: {np.mean(ham_distance_list)} (+- {np.std(ham_distance_list)})')
    print(f'> auc: {np.mean(auc_list)} (+- {np.std(auc_list)})')

    print('------------------------------------------------------------------------')

    # 返回每折的平均值和标准差 标准差的平方 是 方差
    return {'acc_mean':np.round(np.mean(acc_list),4),
            'kappa_mean':np.round(np.mean(kappa_list),4),
            'ham_mean':np.round(np.mean(ham_distance_list),4),
            'auc_mean':np.round(np.mean(auc_list),4),
            'acc_std':np.round(np.std(acc_list),4),
            'kappa_std':np.round(np.std(kappa_list),4),
            'ham_std':np.round(np.std(ham_distance_list),4),
            'auc_std':np.round(np.std(auc_list),4)
            }


