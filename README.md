

 The dataset used by the model is in the datasets folder
 
 程序需要的python版本为3.7
 # 安装依赖
 ```
 pip3 install -r requirements.txt
 ```
 
 # 训练预测模型
 ## 需要改进的地方
 模型搭建的时候有盐刺激浓度，但是训练的时候没有。
 盐刺激浓度可以根据标签转换。输入的数据是有标签的
 ### 解决思路
 
 - 模型迁移。训练好0mm盐刺激的model。继续训练50mm刺激的 model
 
 ```
cd SLSTM-TCNN

# save model and result
mkdir model && mkdir result 

python3 pred_train.py
 ```
 ## 使用训练好的模型
 
 ### LD_DSCP_666
 
 用SLSTM-TCNN扩充DK_DSC_180得到DK_DSCP_666
 
 扩充方法：用DK_DS2_54里的a波形预测9个盐浓度刺激下的b波形。180+54*9=666
 
 扩充的数据集是不知道质量的，因为没有标准的输出。经过测试，相同输入和模型输出也一样。
 
 ### 程序实现
 
 ```
python3 pred_use.py


```


