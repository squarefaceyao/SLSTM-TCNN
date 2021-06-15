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

def kfold_STCM_SLSTM_TCNN(inputs, targets, salt, epochs, datasets, save_model):
    figure_savePath = os.getcwd() + '/figure/STCM_SLSTM_TCNN/{}/'.format(localtime)
    if os.path.exists(figure_savePath) == True:
        print('STCM_SLSTM_TCNN figure save path:"{}"'.format(figure_savePath))
    else:
        os.makedirs(figure_savePath)
        print('STCM_SLSTM_TCNN figure save path::"{}"'.format(figure_savePath))

    num_folds = int(math.log(141)) + 1
    acc_per_fold, auc_per_fold, sen_per_fold = [], [], []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    epochs = epochs
    salt_value = ['0mM', '50mM', '100mM', '150mM', '200mM', '250mM', '300mM', '350mM', '400mM']
    kfold = KFold(n_splits=num_folds, shuffle=True)
    # K-fold Cross Validation model evaluation
    fold_no = 1
    plt.figure(figsize=(15, 12))
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
                            validation_split=0.2)  # ephos=1000 容易过拟合
        # 模型预测
        y_pred = model.predict(inputs[test])

        # 测试集的评价指标

        acc, auc_tf, sen = eva_2_class(l=y_pred, test_y=targets[test], datasets=datasets,
                                       figure_savePath=figure_savePath)

        if save_model == 1:
            model_savePath = os.getcwd() + '/model/STCM_SLSTM_TCNN/{}/'.format(localtime)
            if os.path.exists(model_savePath) == True:
                print('STCM_SLSTM_TCNN save path:"{}"'.format(model_savePath))
            else:
                os.makedirs(model_savePath)
                print('STCM_SLSTM_TCNN save path::"{}"'.format(model_savePath))
            model.save(model_savePath + f'{fold_no}kflod_acc_{acc}.h5')
            print('已经保存模型')
        else:
            print('没有保存模型')

        fpr, tpr, thresholds = roc_curve(targets[test][:, 1], y_pred[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='%s ROC fold %d(area=%0.2f)' % (salt_value[salt], fold_no, roc_auc))

        # 保存每一折里面的评价指标。
        acc_per_fold.append(acc)
        auc_per_fold.append(auc_tf)
        sen_per_fold.append(sen)
        # Increase fold number
        fold_no = fold_no + 1
    # 绘制平均roc
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(tprs, axis=0)

    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area=%0.2f)' % mean_auc,
             lw=3, alpha=.8)
    std_tpr = np.std(tprs, axis=0)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(figure_savePath + '{}_{}_{}.png'.format('CNN', salt_value[salt], localtime), dpi=400)  # CNN+SLSTM-TCNN
    plt.close()

    # == Provide average acc ==
    # 打印指定盐浓度K折交叉验证的训练效果
    aaa = ['0mM', '50mM', '100mM', '150mM', '200mM', '250mM', '300mM', '350mM', '400mM']
    print('盐刺激浓度为:{:.4}'.format(aaa[salt]))
    print('acc per fold')
    for i in range(0, len(acc_per_fold)):
        print('> Fold {} - acc: {:.4} %'.format(i + 1, acc_per_fold[i]))
    print('Average scores for all folds:')
    print('> acc: {:.4} (+- {:.4})'.format(np.mean(acc_per_fold), np.std(acc_per_fold)))
    print(f'计算的平均auc{mean_auc}')
    return {'acc_mean': np.round(np.mean(acc_per_fold), 4),
            'auc_mean': np.round(mean_auc, 4),
            'sen_mean': np.round(np.mean(sen_per_fold), 4),
            'acc_std': np.round(np.std(acc_per_fold), 4),
            'auc_std': np.round(np.std(aucs), 4),
            'sen_std': np.round(np.std(sen_per_fold), 4),

            }

