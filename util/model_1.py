import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def model_3(units,input_size):
    # 预测使用的
    # pcc = 0.850225693960388
    model = keras.Sequential(
        [
            keras.Input(shape=(input_size,1)),
            layers.BatchNormalization(beta_initializer='zero',gamma_initializer='one',name = "nor1"),
            layers.MaxPooling1D(pool_size=2,name='Max1'),
            layers.Conv1D(10, kernel_size=3, strides=2,activation="relu",padding='same',name='Conv1'),
            layers.LSTM(units,return_sequences=True,name='lstm_1'),
            layers.UpSampling1D(size=2,name='TC1'),
            layers.Flatten(name='flatten'),
            layers.Dense(300),
            layers.Dropout(0.5),
            layers.Dense(input_size),
        ]
    )
    return model


def salt(s):
    new_array  = np.zeros((1,10))
    # print('输入的盐浓度是{}mM'.format(s))
    for i in range(1):
        new_array[i] = np.array([0.25]*10)
    new_array = new_array[np.newaxis,:]
    xx2 = layers.Dense(10)(new_array)
    return xx2

def model_4(units,input_size):
    xx2 = salt(s=50)
    input2 = keras.Input(shape=(input_size,1),name='input')
    s = layers.BatchNormalization(beta_initializer='zero',gamma_initializer='one',name = "nor1")(input2)
    s = layers.MaxPooling1D(pool_size=2,name='Max1')(s)
    s = layers.Conv1D(10, kernel_size=3, strides=2,activation="relu",padding='same',name='Conv1')(s)
#     s = layers.Concatenate()([s,xx2])
    s = layers.Multiply()([s,xx2])
    s = layers.LSTM(units,return_sequences=True,name='lstm_1')(s)
    s = layers.UpSampling1D(size=2,name='TC1')(s)
    s = layers.Flatten(name='flatten')(s)
    s = layers.Dense(300)(s)
    s = layers.Dropout(0.5)(s)
    output2 = layers.Dense(input_size)(s)
    model = keras.Model(inputs=[input2], outputs=[output2])
    return model




# def model_3_588To1176(units,input_size):
#     # pcc = 0.850225693960388
#     model = keras.Sequential(
#         [
#             keras.Input(shape=(input_size,1)),
#             layers.BatchNormalization(beta_initializer='zero',gamma_initializer='one',name = "nor1"),
#             layers.MaxPooling1D(pool_size=2,name='Max1'),
#             layers.Conv1D(10, kernel_size=3, strides=2,activation="relu",padding='same',name='Conv1'),
#             layers.LSTM(units,return_sequences=True,name='lstm_1'),
#             layers.UpSampling1D(size=2,name='TC1'),
#             layers.Flatten(name='flatten'),
#             layers.Dense(300),
#             layers.Dropout(0.5),
#             layers.Dense(input_size*2),
#         ]
#     )
#     return model

def cnn_model(input_size):
    # 9 分类使用
    r = "relu"
    ks = 5
    st = 3
    a = 0.005
    model = keras.Sequential(
        [
            keras.Input(shape=(input_size,1)),
            layers.BatchNormalization(beta_initializer='zero',gamma_initializer='one',name = "nor1"),
            layers.Conv1D(10, kernel_size=ks, strides=st,activation=r,padding='same',name='Conv1'),
            layers.MaxPooling1D(pool_size=2,name='Max1'),

            layers.BatchNormalization(beta_initializer='zero',gamma_initializer='one',name = "nor2"),
            layers.Conv1D(10, kernel_size=ks, strides=st,activation=r,padding='same',name='Conv2'),
            layers.MaxPooling1D(pool_size=2,name='Max2'),

            layers.Flatten(name='flatten'),
            layers.Dense(100,kernel_initializer='random_uniform',activation=r,activity_regularizer=keras.regularizers.l2(a),name = "Den1"),
            layers.Dropout(0.2),
            layers.Dense(9,activation="softmax",name = "Den2"),
        ]
    )
    
    model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=['accuracy'])
    return model

def cnn_model_2(input_size):
    # 2分类使用
    r = "relu"
    ks = 5
    st = 3
    a = 0.005
    model = keras.Sequential(
        [
            keras.Input(shape=(input_size,1)),
            layers.BatchNormalization(beta_initializer='zero',gamma_initializer='one',name = "nor1"),
            layers.Conv1D(10, kernel_size=ks, strides=st,activation=r,padding='same',name='Conv1'),
            layers.MaxPooling1D(pool_size=2,name='Max1'),

            # layers.BatchNormalization(beta_initializer='zero',gamma_initializer='one',name = "nor2"),
            # layers.Conv1D(10, kernel_size=ks, strides=st,activation=r,padding='same',name='Conv2'),
            # layers.MaxPooling1D(pool_size=2,name='Max2'),

            # layers.BatchNormalization(beta_initializer='zero',gamma_initializer='one',name = "nor3"),
            # layers.Conv1D(10, kernel_size=ks, strides=st,activation=r,padding='same',name='Conv3'),
            # layers.MaxPooling1D(pool_size=2,name='Max3'),

            # layers.Conv1D(10, kernel_size=3, strides=2,activation="relu",padding='same',name='Conv1'),
            # layers.UpSampling1D(size=2,name='TC1'),
            layers.Flatten(name='flatten'),
            layers.Dense(100,kernel_initializer='random_uniform',activation=r,activity_regularizer=keras.regularizers.l2(a),name = "Den1"),
            layers.Dense(50,kernel_initializer='random_uniform',activation=r,activity_regularizer=keras.regularizers.l2(a),name = "Den2"),

            layers.Dense(10,kernel_initializer='random_uniform',activation=r,activity_regularizer=keras.regularizers.l2(a),name = "Den3"),

            layers.Dropout(0.2),
            layers.Dense(2,activation="softmax",name = "Den4"),
        ]
    )
    
    model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=['accuracy'])
    return model

if __name__=='__main__':
    
    units=12
    
    # model = model_4(units=units,input_size=588)
    model = cnn_model_2(input_size=1176)
    
    # model = cnn_model(input_size=1176)
    # model4 = model_4(12,588)
    model.summary()
