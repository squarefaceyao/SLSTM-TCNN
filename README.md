
# A deep learning method for the long-term prediction of plant electrical signals under salt stress to identify salt tolerance


 ## Citation

Please cite the following work if you find the data/code useful.

```
@article{yao2021,
  title={A deep learning method for the long-term prediction of plant electrical signals under salt stress to identify salt tolerance},
  author={Yang, Carl and Xiao, Yuxin and Zhang, Yu and Sun, Yizhou and Han, Jiawei},
  journal={COMPAG},
  year={2021}
}
```
## Contact

Please contact us if you have problems with the data/code, and also if you think your work is relevant but missing from the survey.

Jiepeng Yao(squarefaceyao@gmail.com)


 The dataset used by the model is in the datasets folder
 
 ## Guideline
 ### Stage 1: Installation package
 ```
 pip3 install -r requirements.txt
 ```
### Stage 2: Training
 ```
cd SLSTM-TCNN

# save model and result
mkdir model && mkdir result 

python3 pred_train.py
 ```
### Stage 3: Predicting
 
### Predict wave b
  
 ```
python3 pred_use.py

```

#### SCDM

```
python3 SCDM+SLSTM_TCNN.py

```
#### STCM

```
python3 STCM+SLSTM-TCNN.py

```

