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


def kfold(inputs, targets,wheat,salt):
    num_folds = 3
    pcc_per_fold = []
    fre_per_fold = []
    mse_per_fold = []
    epochs=600
    kfold = KFold(n_splits=num_folds, shuffle=True)
    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(inputs, targets):
        # 加载模型
        units=12# 78、6、12 效果最好
        # 预测-model_3效果最好
        print(inputs[train].shape)
        print(inputs[test].shape)
        model = model_3(units=units,
                        input_size=inputs[train].shape[1]) 
        model.compile(loss='mean_squared_error',
                      optimizer='Adam', 
                      metrics=['mae'])
        
          # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')  
        # 模型训练
        # verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
        history = model.fit(inputs[train], 
                            targets[train], 
                            epochs=epochs,
                            verbose=0,
                            validation_split=0.1) # ephos=1000 容易过拟合
        
        #   打印模型训练时的评价指标
        print('在验证集上的平均mea是:{:.4}'.format(np.mean(history.history['val_mae'])))
         # Generate generalization metrics
        l = model.predict(inputs[test])
        pp = '{}在{}salt测试集上的数量={}'.format(wheat,salt,l.shape[0])
        print('\033[1;37;43m   {}!\033[0m'.format(pp))
        # 测试集的皮尔逊系数
        pcc = []
        for i in range(0,targets[test].shape[0]):
            pccs = pearsonr(targets[test][i],l[i])
            pcc.append(pccs[0])
        ave_pcc = np.mean(pcc)
        # 测试集的frechet
        frechet = []
        for i in range(0,targets[test].shape[0]):
            fr = get_similarity(targets[test][i],l[i])
            frechet.append(fr)
        ave_fre = np.mean(frechet)
        # 测试集的rmse
        rmse = []
        for i in range(0,targets[test].shape[0]):
            rm = np.sqrt(mean_squared_error(targets[test][i],l[i]))
            rmse.append(rm)
        ave_mse = np.mean(rmse)
        # 保存每一折里面的评价指标。
        pcc_per_fold.append(ave_pcc)
        fre_per_fold.append(ave_fre)
        mse_per_fold.append(ave_mse)
        # Increase fold number
        fold_no = fold_no + 1

    # == Provide average pcc fre mse ==
    # 打印指定盐浓度K折交叉验证的训练效果
    aaa = ['0mM','50mM','100mM','150mM','200mM','250mM','300mM','350mM','400mM']
    print('预测的小麦品种为{},盐刺激浓度为:{}'.format(wheat,aaa[salt]))  
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

#k_cnn_2 使用的交叉验证

def cover(test_y):
    test = []
    for i in range(test_y.shape[0]):
        test.append(argmax(test_y[i]))
    yy = np.array(test)
    yy2 = np.squeeze(yy)
    return yy2

def kfold_cnn_2(inputs, targets,salt,epochs,datasets,save_model):
    figure_savePath = os.getcwd() + '/figure/STCM/{}/'.format(localtime)
    if os.path.exists(figure_savePath) == True:
        print('STCM figure save path:"{}"'.format(figure_savePath))
    else:
        os.makedirs(figure_savePath)
        print('STCM figure save path::"{}"'.format(figure_savePath))


    num_folds = int(math.log(141))+1
    acc_per_fold,auc_per_fold,sen_per_fold = [],[],[]
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0,1,100)

    epochs=epochs
    salt_value = ['0mM','50mM','100mM','150mM','200mM','250mM','300mM','350mM','400mM']
    kfold = KFold(n_splits=num_folds, shuffle=True)
    # K-fold Cross Validation model evaluation
    fold_no = 1
    plt.figure(figsize=(15,12))
    sns.set(font_scale=3)
    for train, test in kfold.split(inputs, targets):
        
        print(f'测试集的数量是{inputs[test].shape[0]}')
        
        # 预测-model_3效果最好
        model = cnn_model_2(input_size=1176)
        # Generate a print
        # 模型训练
        # verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
        history = model.fit(inputs[train], 
                            targets[train], 
                            epochs=epochs,
                            verbose=0,
                            validation_split=0.2) # ephos=1000 容易过拟合
        # 模型预测
        y_pred = model.predict(inputs[test])

        # 测试集的评价指标

        acc,auc_tf,sen = eva_2_class(l=y_pred,test_y=targets[test],datasets=datasets,figure_savePath=figure_savePath)

        if save_model == 1:
            model_savePath = os.getcwd()+'/model/STCM/{}/'.format(localtime)
            if os.path.exists(model_savePath) == True:
                print('STCM save path:"{}"'.format(model_savePath))
            else:
                os.makedirs(model_savePath)
                print('STCM save path::"{}"'.format(model_savePath))
            model.save(model_savePath + f'{fold_no}kflod_acc_{acc}.h5')
            print('已经保存模型')
        else:
            print('没有保存模型')

        fpr, tpr, thresholds  =  roc_curve(targets[test][:,1], y_pred[:,1]) 
        tprs.append(interp(mean_fpr,fpr,tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr,tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr,lw=2, alpha=0.3, label='%s ROC fold %d(area=%0.2f)'% (salt_value[salt],fold_no,roc_auc))
       
        # 保存每一折里面的评价指标。
        acc_per_fold.append(acc)
        auc_per_fold.append(auc_tf)
        sen_per_fold.append(sen)
        # Increase fold number
        fold_no = fold_no + 1
    # 绘制平均roc
    mean_tpr = np.mean(tprs,axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(tprs, axis=0)

    plt.plot(mean_fpr, mean_tpr,color='b',label = r'Mean ROC (area=%0.2f)'%mean_auc,
                                           lw=3,alpha=.8)
    std_tpr = np.std(tprs,axis=0)

    plt.xlim([-0.05,1.05])
    plt.ylim([-0.05,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(figure_savePath+'{}_{}_{}.png'.format('CNN',salt_value[salt],localtime),dpi=400) # CNN+SLSTM-TCNN
    plt.close()

    # == Provide average acc ==
    # 打印指定盐浓度K折交叉验证的训练效果
    aaa = ['0mM','50mM','100mM','150mM','200mM','250mM','300mM','350mM','400mM']
    print('盐刺激浓度为:{:.4}'.format(aaa[salt]))  
    print('acc per fold')
    for i in range(0, len(acc_per_fold)):
        print('> Fold {} - acc: {:.4} %'.format(i+1,acc_per_fold[i]))
    print('Average scores for all folds:')
    print('> acc: {:.4} (+- {:.4})'.format(np.mean(acc_per_fold),np.std(acc_per_fold)))
    print(f'计算的平均auc{mean_auc}')
    return {'acc_mean':np.round(np.mean(acc_per_fold),4),
            'auc_mean':np.round(mean_auc,4),
            'sen_mean':np.round(np.mean(sen_per_fold),4),
            'acc_std':np.round(np.std(acc_per_fold),4),
            'auc_std':np.round(np.std(aucs),4),
            'sen_std': np.round(np.std(sen_per_fold), 4),

            }


def kfold_cnn_9(inputs, targets,epochs,wheat,save_model):

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
            model_savePath = 'model/{}_SCDM/{}/'.format(wheat, localtime)
            if os.path.exists(model_savePath) == True:
                print('SCDM save path:"{}"'.format(model_savePath))
            else:
                os.makedirs(model_savePath)
                print('SCDM save path::"{}"'.format(model_savePath))
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


