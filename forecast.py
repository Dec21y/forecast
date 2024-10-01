import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import tensorflow as tf
tf.random.set_seed(2)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_squared_error
from math import sqrt

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 设置定量的GPU使用量

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存
session = tf.compat.v1.Session(config=config)

# 设置最小的GPU使用量

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
np.set_printoptions(threshold=sys.maxsize)

'''
 数据导入
'''
# start = time.perf_counter()

df_P=pd.read_csv("pre2.csv")
df_T=pd.read_csv("tas2.csv")
df_Q=pd.read_csv("GRUN2.csv")
df_m=pd.read_csv("gridselete.csv",header=None)
# for j in range(df_P.shape[1]):
for m in (df_m.iloc[1162:1185,0]):
    print("m", m)
    path1 = "grid parameter"
    path2 = str(m)
    path3 = ".csv"
    path4="F:/新建文件夹/参数/"
    path = path4+path1 + path2 + path3
    print(path)
    canshuvalue = pd.read_csv(path, header=1)
    canshuvalue = canshuvalue.values
    canshuvalue = np.array(canshuvalue)

    batch_sizevalue = canshuvalue[0, 9]
    epochsvalue = canshuvalue[0, 10]
    nerousvalue = canshuvalue[0, 12]
    print(batch_sizevalue,epochsvalue,nerousvalue)

    value=[]
    df_Tm=df_T.iloc[:, m]
    Tm = np.array(df_Tm)
    Tm = Tm.reshape(-1, 1)
    df_Qm=df_Q.iloc[:, m]
    Qm = np.array(df_Qm)
    Qm = Qm.reshape(-1, 1)
    df_Pm=df_P.iloc[:, m]
    Pm = np.array(df_Pm)
    Pm = Pm.reshape(-1, 1)
    value.extend((Qm,Pm,Tm))
    value=np.array(value)
    value=value.reshape(-1,1356)
    value=value.transpose()
    value=pd.DataFrame(value)
    outputpath='A1.csv'
    value.to_csv(outputpath,sep=',',index=False,header=False)
    """数据划分 """
    df= pd.read_csv("A1.csv",header=None)
    df = df.values
    print(df.shape)
    test_split = round(len(df) * 0.20)
    df_for_training = df[:-int(len(df) * 0.20)]
    df_for_testing = df[-int(len(df) * 0.20):]
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_for_training_scaled = scaler.fit_transform(df_for_training)
    df_for_testing_scaled = scaler.transform(df_for_testing)

    """滞后时长确定"""
    x = np.linspace(0, 40, df.shape[0])  # 图像仅在0到20之间显示，显示1356个
    y1 = list(df[:,0])
    y2 = list(df[:,1])
    y3 = list(df[:,2])
    """利用pearson计算滞后性"""
    # 径流与降雨
    data_cor1 = pd.DataFrame(np.array([y1, y2]).T, columns=['y1', 'y2'])
    for i in range(0, 40):
        data_cor1[str(i)] = data_cor1['y2'].shift(i)
    data_cor1.dropna(inplace=True)
    p = data_cor1.corr()
    # print("person相关系数：\n", data_cor1.corr())
    Q_P=data_cor1.corr().iloc[0][2:].values.tolist().index(max(data_cor1.corr().iloc[0][2:].values))
    print(Q_P)
    # print(data_cor1.corr().iloc[0][2:].values.tolist().index(max(data_cor1.corr().iloc[0][2:].values)))
    # 径流与温度
    data_cor2 = pd.DataFrame(np.array([y1, y3]).T, columns=['y1', 'y3'])
    for i in range(0, 40):
        data_cor2[str(i)] = data_cor2['y3'].shift(i)
    data_cor2.dropna(inplace=True)
    p = data_cor2.corr()
    # print("person相关系数：\n", data_cor2.corr())
    Q_T = data_cor2.corr().iloc[0][2:].values.tolist().index(max(data_cor2.corr().iloc[0][2:].values))
    print(Q_T)
    # 径流与径流
    cor = []
    for i in range(1, 40, 1):
        y4 = list(df[i:, 0])
        y4 = np.array(y4).reshape(1, -1)
        y5 = list(df[:-i, 0])
        y5 = np.array(y5).reshape(1, -1)
        # print(y5)
        y4_y5 = np.r_[y4, y5]
        c_y4_y5 = np.corrcoef(y4_y5)[0, 1]
        cor.append(c_y4_y5)
    # print(cor)
    value = max(cor)
    Q_Q=cor.index(value) + 1
    print(Q_Q)
    n_past_dis=Q_Q
    n_past_pr=Q_P
    n_past_T=Q_T
    def createXY(dataset,n_past_dis,n_past_pr,n_past_T):
        dataX = []
        dataY = []
        for w in range(max(n_past_dis,n_past_pr,n_past_T), len(dataset)):
            n = []
            a = dataset[w - n_past_dis:w, 0]
            n.extend(a)
            b = dataset[w - n_past_pr:w+1, 1]
            n.extend(b)
            c = dataset[w - n_past_T:w + 1, 2]
            n.extend(c)
            n=np.array(n).reshape(1,Q_Q+Q_P+Q_T+2)
            dataX.append(n)
            dataY.append(dataset[w, 0])
        return np.array(dataX), np.array(dataY)

    trainX,trainY=createXY(df_for_training_scaled,Q_Q,Q_P,Q_T)
    testX,testY=createXY(df_for_testing_scaled,Q_Q,Q_P,Q_T)

    score = {}
    def build_model(q, optimizer):
        grid_model = Sequential()
        grid_model.add(LSTM(q,return_sequences=True,input_shape=(1,Q_Q+Q_P+Q_T+2)))
        #activation='relu',
        grid_model.add(LSTM(q))
        grid_model.add(Dropout(0.1))
        grid_model.add(Dense(1))
        grid_model.compile(loss = 'mse',optimizer = optimizer)
        return grid_model
    grid_model =KerasRegressor(build_fn=build_model,verbose=1,validation_data=(testX,testY))
    parameters = {'batch_size': range(batch_sizevalue, batch_sizevalue+1, 1),
                      'epochs': range(epochsvalue, epochsvalue+1, 1),
                      'q': range(nerousvalue,nerousvalue+1,1),
                      'optimizer': ['adam']}
    grid_search  = GridSearchCV(estimator = grid_model,
                            param_grid = parameters,
                            cv = 2)
    grid_search = grid_search.fit(trainX,trainY)
    grid_search.best_params_
    my_model=grid_search.best_estimator_.model

    # dir_name = 'D:/pycharm代码/1/build model data'
    dir_name = '.'
    model="LSTM_model" + str(m) + ".h5"
    my_model.save(dir_name+"/"+model)
    """
    # my_model.train()
    prediction = my_model.predict(testX)
    # prediction=yuce2-my_model1.h5.predict(testX)
    # print("prediction\n", prediction)
    # print("\nPrediction Shape-",prediction.shape)
    prediction_copies_array = np.repeat(prediction,3, axis=-1)
    pred1=scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction),3)))[:,0]
    original_copies_array = np.repeat(testY,3, axis=-1)
    original1=scaler.inverse_transform(np.reshape(original_copies_array,(len(testY),3)))[:,0]


    # 实测值与预测值图
    # print("Pred Values-- " ,pred1)
    print(pred1.shape)
    # print("\nOriginal Values-- " ,original1)
    print(original1.shape)
    # with open("result1698-1698（换）.txt", 'a') as f:
    #     f.write(str(pred1))
    #     f.write('\n')
    #     # f.write(str(original1))
    #     # f.write('\n')
    #     # f.write('\n')
    r2 = r2_score(original1, pred1)
    rmse_test = sqrt(mean_squared_error(original1, pred1))
    mae_test = mean_absolute_error(original1, pred1)
    mse_test = mean_squared_error(original1, pred1)

    # 训练集
    prediction=my_model.predict(trainX)
    prediction_copies_array = np.repeat(prediction,3, axis=-1)
    pred2=scaler.inverse_transform(np.reshape(prediction_copies_array,(len(prediction),3)))[:,0]
    original_copies_array = np.repeat(trainY,3, axis=-1)
    original2=scaler.inverse_transform(np.reshape(original_copies_array,(len(trainY),3)))[:,0]
    r2_train = r2_score(original2, pred2)
    rmse_train = sqrt(mean_squared_error(original2, pred2))
    mae_train = mean_absolute_error(original2, pred2)
    mse_train = mean_squared_error(original2, pred2)

    score['m'] = m
    score['r2'] = r2
    score['r2_train'] = r2_train
    score['rmse_test'] = rmse_test
    score['rmse_train'] = rmse_train
    score['mae_train'] = mae_train
    score['mae_test'] = mae_test
    score['mse_train'] = mse_train
    score['mse_test'] = mse_test

    parameters = grid_search.best_params_
    bs = parameters.get('batch_size')
    epo = parameters.get('epochs')
    opy = parameters.get('optimizer')
    ner = parameters.get('q')
    score['batch_size'] = bs
    score['epochs'] = epo
    score['optimizer'] = opy
    score['q'] = ner
    #score |= grid_search.best_params_
    df_table = pd.DataFrame(list(score.items())).T
    print(df_table)
    df_table.to_csv("./grid parameter"+str(m)+".csv",index=False)
    """
    max1 = -max(n_past_dis, n_past_pr, n_past_T)
    df = pd.read_csv("A1.csv", header=None)
    df_5_days_past = df.iloc[max1:, :]
    df_5_days_past.columns = ["Q", "P", "T"]

    # df_pr = pd.read_csv("CanESM5-ssp126-prmonthly.csv")
    df_pr = pd.read_csv("CanESM5-ssp126-pr栅格乘面积.csv")
    # print(df_P)
    df_tem = pd.read_csv("CanESM5-ssp126-tasmonthly.csv")
    value = []

    df_pr = df_pr.iloc[:, m]
    df_pr = np.array(df_pr)
    pr = df_pr.reshape(-1, 1)
    # print("pr",pr)

    df_tem = df_tem.iloc[:, m]
    df_tem = np.array(df_tem)
    tem = df_tem.reshape(-1, 1)
    print("tem", tem.shape)

    value.extend((pr, tem))
    # print("value",value)
    value = np.array(value)
    value = value.reshape(-1, 1020)
    value = value.transpose()
    value = pd.DataFrame(value)

    value.columns = ["P", "T"]
    value = value.reindex(columns=list('QPT'), fill_value=" ")

    full_df = pd.concat([pd.DataFrame(df_5_days_past), value]).reset_index().drop(["index"], axis=1)
    outputpath = 'B1.csv'
    full_df.to_csv(outputpath, sep=',', index=False, header=False)
    full_df = full_df.apply(pd.to_numeric, errors='coerce')
    full_df_scaled_array = full_df.values.astype('float32')

    all_data = []
    time_step = -max1
    print('time_step', time_step)

    for i in range(time_step, len(full_df_scaled_array) - 648):
        data_x = []
        a = full_df_scaled_array[i - n_past_dis:i, 0:1]
        a = [i for item in a for i in item]
        a = np.array(a)
        data_x.append(a)

        b = full_df_scaled_array[i - n_past_pr:i + 1, 1:2]
        b = [i for item in b for i in item]
        b = np.array(b)
        data_x.append(b)

        c = full_df_scaled_array[i - n_past_T:i + 1, 2:3]
        c = [i for item in c for i in item]
        c = np.array(c)
        data_x.append(c)

        data_x = tuple(data_x)
        data_x = np.hstack(data_x)

        data_x = pd.DataFrame(data_x)

        data_x = data_x.to_numpy().reshape((1, n_past_dis + n_past_pr + n_past_T + 2, 1)).astype('float32')

        data_x = np.transpose(data_x, (0, 2, 1))
        # print("data_x", data_x)
        # print("data_x",data_x)

        prediction = my_model.predict(data_x)
        # print("prediction",prediction)
        full_df_scaled_array = pd.DataFrame(full_df_scaled_array)
        full_df_scaled_array.loc[i, 0] = prediction
        full_df_scaled_array = full_df_scaled_array.to_numpy().astype('float32')
        all_data.append(prediction)
        new_array = np.array(all_data)
        new_array = new_array.reshape(-1, 1)
        # print(new_array)
        new_array = pd.DataFrame(new_array)
        new_array.to_csv("./预测(201501-204512)" + '-' + str(m) + ".csv", index=False, header=False)
