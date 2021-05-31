import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# 输入（None,1176）输出（N,174）

def pick_arange(arange, num):
    # 等距离取数，num为数据长度。
    if num > len(arange):
        # print("# num out of length, return arange:", end=" ")
        return arange
    else:
        output = np.array([], dtype=arange.dtype)
        seg = len(arange) / num
        for n in range(num):
            if int(seg * (n+1)) >= len(arange):
                output = np.append(output, arange[-1])
            else:
                output = np.append(output, arange[int(seg * n)])
#         print("# return new arange:", end=' ')
        return output
def data_dispose(df_T):
    # 原始波形切分，分出x和y
    # df_T = df_T.T
    b=int(df_T.shape[1]/2)
    xx1 = df_T.iloc[:,0:b]
    yy1 = df_T.iloc[:,b:df_T.shape[1]]    
    # label =df_T.iloc[:,b*2:df_T.shape[1]] #类标签
    xx = np.array(xx1)/100
    yy = np.array(yy1)/100
    xx_ = xx.reshape(len(xx),xx.shape[1],1) # 转换成三个维度
    yy_ = yy.reshape(len(yy),yy.shape[1]) # 转换成二个维度
    return xx_,yy_

def data_ultimately(df_T,num,test_size,random_state):
    # 最终处理，包含等距离取数，切分训练集
    test={} # 创建字典，保存转换结果
    for i in range(df_T.shape[0]):
        ccc = np.array(df_T.iloc[i]) # 使用iloc切片不会出现问题。处理的数据是（None,1176）
        test[i]=pick_arange(ccc,num) # 保存转换后的数据到字典
        del ccc # 删除每一个循环里的变量
    bbb = pd.DataFrame(test,index=None).T 
    xx_,yy_ = data_dispose(bbb) # 调用切分数据函数
    train_x,test_x,train_y,test_y = train_test_split(xx_, yy_, test_size=test_size, random_state=random_state) # 切分数据集
    # 这里把标签放到y后面，随机划分数据集后标签还是在y后面跟着，通过切片分出label
    train_label = train_y[: ,-1]*100.0
    test_label = test_y[: ,-1]*100.0
    # 除去最后一列都是b波
    train_y = train_y[:, :-1]
    test_y = test_y[:, :-1]
        # label 需要转换成int类型
    return train_x,test_x,train_y,test_y,train_label.astype(int),test_label.astype(int) 

if __name__=='__main__':
    data_ultimately(df_T,num,test_size,random_state)
