# -*- coding: utf-8 -*-
from __future__ import division
from sklearn.metrics import r2_score,explained_variance_score
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn import svm,ensemble,preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import time as time
import gc

# 读取数据集
def read_old_test():
    attribute_list = ['eage_num', 'max_lambda', 'L','node_num']
    df_test = pd.read_csv('tvshow_old.csv')                             # 以边数和最大特征值作为特征的测试集
    X_test = df_test.loc[:, attribute_list].values
    y_test = df_test.loc[:, 't'].values
    return X_test,y_test
def read_new_test():
    Ndf_test = pd.read_csv('tvshow_new.csv')                              # 以邻接矩阵作为特征学习比值的测试集
    NX_test = Ndf_test.values[:, 1:2501].astype(np.int)
    ny_test = Ndf_test.values[:, 2501]
    return NX_test,ny_test
def read_old():
    attribute_list = ['eage_num', 'max_lambda', 'L','node_num']
    df = pd.read_csv('30_government.csv')                              # 以边数和最大特征值作为特征学习比值的训练集
    df = df.append(pd.read_csv('40_politician.csv'))
    X = df.loc[:, attribute_list].values
    y = df.loc[:, 't'].values
    return X,y
def read_new():
    Ndf = pd.read_csv('government_new.csv')                              # 以邻接矩阵作为特征学习比值的训练集
    Ndf = Ndf.append(pd.read_csv('politician_new.csv'))
    NX = Ndf.values[:, 1:2501].astype(np.int)
    ny = Ndf.values[:, 2501]
    return  NX,ny

def svm_fit_model_k_fold(X, y):
   regressor = svm.SVR()
   params = {'kernel':('linear','rbf','sigmoid'),'C':[0.8,1,1.2],'gamma':[0.2,0.5,1,1.5]}
   grid = GridSearchCV(regressor, param_grid=params,scoring='r2',cv=5,n_jobs=3,pre_dispatch=3)
   grid = grid.fit(X, y)
   return grid

def mlp_fit_model_k_fold(X,y):
   regressor = MLPRegressor(solver='lbfgs', alpha=1e-5,random_state=1)
   mytuple = []
   for j in range(5,20,5):
      for i in range(2,5):
         ituple = (j,) * i
         mytuple.append(ituple)
   mytuple.append((5,4,3))
   mytuple.append((3,4,5))
   mytuple.append((2,4,6))
   mytuple.append((6,4,2))
   mytuple.append((10,7,4,2))
   mytuple.append((2,4,7,10))
   mytuple.append((10,6,3))
   mytuple.append((3,6,10))
   params = {'learning_rate': ('invscaling', 'adaptive'), 'hidden_layer_sizes':mytuple,'activation':('relu','tanh','logistic') }
   grid = GridSearchCV(regressor, param_grid=params, scoring='r2', cv=5, n_jobs=3,pre_dispatch=3)
   grid = grid.fit(X, y)
   return grid

def rf_fit_model_k_fold(X,y):
    regressor = ensemble.RandomForestRegressor(random_state=1)
    params = {'n_estimators': [50,80,100,120,150], 'max_features':('sqrt','log2'), 'max_depth': [20,50,100]}
    grid = GridSearchCV(regressor, param_grid=params, scoring='r2', cv=5, n_jobs=3, pre_dispatch=3)
    grid = grid.fit(X, y)
    return grid

def rf_fit_model_k_fold_old(X,y):
    regressor = ensemble.RandomForestRegressor(random_state=1,max_features=None)
    params = {'n_estimators': [50,80,100,120,150]}
    grid = GridSearchCV(regressor, param_grid=params, scoring='r2', cv=5, n_jobs=3, pre_dispatch=3)
    grid = grid.fit(X, y)
    return grid
# svm回归方法
def usesvm():
   # 使用边数和最大特征值进行回归
   X,y = read_old()
   X = preprocessing.MinMaxScaler().fit_transform(X)                                       # 网格搜索特征归一化
   reg = svm_fit_model_k_fold(X, y)                                                            # 搜索最优参数
   clf = svm.SVR(kernel=reg.best_params_['kernel'],C=reg.best_params_['C'],gamma=reg.best_params_['gamma'])
   X_train, X_ver, y_train, y_ver = train_test_split(X, y, test_size=0.2)                    # 划分训练集和验证集
   del X,y
   gc.collect()                                                                              # 释放内存
   clf.fit(X_train,y_train)                                                                  # 训练模型
   predicted = clf.predict(X_ver)                                                           # 在验证集上验证
   # 保存最优学习器配置及验证集表现
   result = np.vstack((y_ver,predicted))
   dataframe = pd.DataFrame({'y_ver': result[0], 'predicted': result[1]})
   dataframe.to_csv("svm_old_ver.csv", index=True, sep=',')
   bestparams = "kernel:" + reg.best_params_['kernel'] + 'C:' + str(reg.best_params_['C']) + "gamma:" + str(reg.best_params_['gamma'])
   fh = open("svm_old_ver.txt", 'a')
   fh.write(bestparams + '\n' + "解释方差：" + str(explained_variance_score(y_ver, predicted))+ '\n' + "r2_score:" + str(r2_score(y_ver,predicted)))
   fh.close()
   del X_train, X_ver, y_train, y_ver
   gc.collect()                                                                              # 释放内存
   # 保存最优学习器测试集表现
   X_test, y_test = read_old_test()
   X_test = preprocessing.MinMaxScaler().fit_transform(X_test)
   predicted_test = reg.predict(X_test)
   result_test = np.vstack((y_test,predicted_test))
   dataframe_test = pd.DataFrame({'y_test': result_test[0], 'predicted': result_test[1]})
   dataframe_test.to_csv("svm_old_test.csv", index=True, sep=',')
   fh = open("svm_old_test.txt", 'a')
   fh.write(bestparams + '\n' + "解释方差：" + str(explained_variance_score(y_test, predicted_test)) + '\n' + "r2_score:" + str(r2_score(y_test, predicted_test)))
   fh.close()
   del X_test, y_test
   gc.collect()
   print('old')
   '''
   # 使用邻接矩阵进行回归
   NX, ny = read_new()
   time_start = time.time()
   Nreg = svm_fit_model_k_fold(NX,ny)
   Nclf = svm.SVR(kernel=Nreg.best_params_['kernel'],C=Nreg.best_params_['C'], gamma=Nreg.best_params_['gamma'])
   NX_train, NX_ver, ny_train, ny_ver = train_test_split(NX, ny, test_size=0.2)                   # 划分训练集和验证集
   del NX,ny
   gc.collect()
   Nclf.fit(NX_train, ny_train)
   Npredicted = Nclf.predict(NX_ver)
   # 保存最优学习器配置及验证集表现
   Nresult = np.vstack((ny_ver, Npredicted))
   Ndataframe = pd.DataFrame({'y_ver': Nresult[0], 'predicted': Nresult[1]})
   Ndataframe.to_csv("svm_new_ver.csv", index=True, sep=',')
   Nbestparams = "kernel:" + Nreg.best_params_['kernel'] + 'C:' + str(Nreg.best_params_['C']) + "gamma:" + str(Nreg.best_params_['gamma'])
   fh = open("svm_new_ver.txt", 'a')
   fh.write(Nbestparams + '\n' + "解释方差：" + str(explained_variance_score(ny_ver, Npredicted)) + '\n' + "r2_score:" + str(r2_score(ny_ver, Npredicted)))
   fh.close()
   del NX_train, NX_ver, ny_train, ny_ver
   gc.collect()
   # 保存最优学习器测试集表现
   NX_test, ny_test = read_new_test()
   Npredicted_test = Nreg.predict(NX_test)
   Nresult_test = np.vstack((ny_test, Npredicted_test))
   Ndataframe_test = pd.DataFrame({'y_test': Nresult_test[0], 'predicted': Nresult_test[1]})
   Ndataframe_test.to_csv("svm_new_test.csv", index=True, sep=',')
   fh = open("svm_new_test.txt", 'a')
   fh.write(Nbestparams + '\n' + "解释方差：" + str(explained_variance_score(ny_test, Npredicted_test)) + '\n' + "r2_score:" + str(r2_score(ny_test, Npredicted_test)))
   fh.close()
   del NX_test, ny_test
   gc.collect()
   time_end = time.time()
   print(time_end-time_start)
   '''
# MLP回归方法
def usemlp():
   # 使用边数和最大特征值进行回归
   X, y = read_old()
   X = preprocessing.MinMaxScaler().fit_transform(X)  # 网格搜索特征归一化
   reg = mlp_fit_model_k_fold(X, y)  # 搜索最优参数
   clf = MLPRegressor(solver='lbfgs',alpha=1e-5,random_state=1,learning_rate=reg.best_params_['learning_rate'],hidden_layer_sizes=reg.best_params_['hidden_layer_sizes'],activation=reg.best_params_['activation'])
   X_train, X_ver, y_train, y_ver = train_test_split(X, y, test_size=0.2)  # 划分训练集和验证集
   del X, y
   gc.collect()  # 释放内存
   clf.fit(X_train, y_train)  # 训练模型
   predicted = clf.predict(X_ver)  # 在验证集上验证
   # 保存最优学习器配置及验证集表现
   result = np.vstack((y_ver, predicted))
   dataframe = pd.DataFrame({'y_ver': result[0], 'predicted': result[1]})
   dataframe.to_csv("mlp_old_ver.csv", index=True, sep=',')
   bestparams = "learning_rate:" + reg.best_params_['learning_rate'] + 'hidden_layer_sizes:' + str(reg.best_params_['hidden_layer_sizes']) + "activation:" + str(reg.best_params_['activation'])
   fh = open("mlp_old_ver.txt", 'a')
   fh.write(bestparams + '\n' + "解释方差：" + str(explained_variance_score(y_ver, predicted)) + '\n' + "r2_score:" + str(r2_score(y_ver, predicted)))
   fh.close()
   del X_train, X_ver, y_train, y_ver
   gc.collect()  # 释放内存
   # 保存最优学习器测试集表现
   X_test, y_test = read_old_test()
   X_test = preprocessing.MinMaxScaler().fit_transform(X_test)
   predicted_test = reg.predict(X_test)
   result_test = np.vstack((y_test, predicted_test))
   dataframe_test = pd.DataFrame({'y_test': result_test[0], 'predicted': result_test[1]})
   dataframe_test.to_csv("mlp_old_test.csv", index=True, sep=',')
   fh = open("mlp_old_test.txt", 'a')
   fh.write(bestparams + '\n' + "解释方差：" + str(explained_variance_score(y_test, predicted_test)) + '\n' + "r2_score:" + str(r2_score(y_test, predicted_test)))
   fh.close()
   del X_test, y_test
   gc.collect()
   print('old')
   '''
   # 使用邻接矩阵作为特征回归
   NX, ny = read_new()
   time_start = time.time()
   Nreg = mlp_fit_model_k_fold(NX,ny)
   Nclf = MLPRegressor(solver='lbfgs',alpha=1e-5,random_state=1,learning_rate=Nreg.best_params_['learning_rate'],hidden_layer_sizes=Nreg.best_params_['hidden_layer_sizes'],activation=Nreg.best_params_['activation'])
   NX_train, NX_ver, ny_train, ny_ver = train_test_split(NX, ny, test_size=0.2)
   del NX, ny
   gc.collect()
   Nclf.fit(NX_train, ny_train)
   Npredicted = Nclf.predict(NX_ver)
   # 保存最优学习配置及验证集表现
   Nresult = np.vstack((ny_ver, Npredicted))
   Ndataframe = pd.DataFrame({'y_ver': Nresult[0], 'predicted': Nresult[1]})
   Ndataframe.to_csv("mlp_new_ver.csv", index=True, sep=',')
   Nbestparams = "learning_rate:" + Nreg.best_params_['learning_rate'] + 'hidden_layer_sizes:' + str(Nreg.best_params_['hidden_layer_sizes']) + "activation:" + str(Nreg.best_params_['activation'])
   fh = open("mlp_new_ver.txt", 'a')
   fh.write(Nbestparams + '\n' + "解释方差：" + str(explained_variance_score(ny_ver, Npredicted)) + '\n' + "r2_score:" + str(r2_score(ny_ver, Npredicted)))
   fh.close()
   del NX_train, NX_ver, ny_train, ny_ver
   gc.collect()
   # 保存最优学习器测试集表现
   NX_test, ny_test = read_new_test()
   Npredicted_test = Nreg.predict(NX_test)
   Nresult_test = np.vstack((ny_test, Npredicted_test))
   Ndataframe_test = pd.DataFrame({'y_test': Nresult_test[0], 'predicted': Nresult_test[1]})
   Ndataframe_test.to_csv("mlp_new_test.csv", index=True, sep=',')
   fh = open("mlp_new_test.txt", 'a')
   fh.write(Nbestparams + '\n' + "解释方差：" + str(explained_variance_score(ny_test, Npredicted_test)) + '\n' + "r2_score:" + str(r2_score(ny_test, Npredicted_test)))
   fh.close()
   del NX_test, ny_test
   gc.collect()
   time_end = time.time()
   print(time_end-time_start)
  '''
# random forest回归方法
def userf():                             # 注意与前种方法不同，随机森林使用旧特征不需要归一化，此外在新老特征上所搜索的参数以及初始条件不同。
    # 使用边数和最大特征值进行回归
    X, y = read_old()
    reg = rf_fit_model_k_fold_old(X, y)  # 搜索最优参数
    clf = ensemble.RandomForestRegressor(max_features=None,random_state=1,n_estimators=reg.best_params_['n_estimators'])
    X_train, X_ver, y_train, y_ver = train_test_split(X, y, test_size=0.2)  # 划分训练集和验证集
    del X, y
    gc.collect()  # 释放内存
    clf.fit(X_train, y_train)  # 训练模型
    predicted = clf.predict(X_ver)  # 在验证集上验证
    # 保存最优学习器配置及验证集表现
    result = np.vstack((y_ver, predicted))
    dataframe = pd.DataFrame({'y_ver': result[0], 'predicted': result[1]})
    dataframe.to_csv("rf_old_ver.csv", index=True, sep=',')
    bestparams = "n_estimators:" + str(reg.best_params_['n_estimators'])
    fh = open("rf_old_ver.txt", 'a')
    fh.write(bestparams + '\n' + "解释方差：" + str(explained_variance_score(y_ver, predicted)) + '\n' + "r2_score:" + str(r2_score(y_ver, predicted)))
    fh.close()
    del X_train, X_ver, y_train, y_ver
    gc.collect()  # 释放内存
    # 保存最优学习器测试集表现
    X_test, y_test = read_old_test()
    predicted_test = reg.predict(X_test)
    result_test = np.vstack((y_test, predicted_test))
    dataframe_test = pd.DataFrame({'y_test': result_test[0], 'predicted': result_test[1]})
    dataframe_test.to_csv("rf_old_test.csv", index=True, sep=',')
    fh = open("rf_old_test.txt", 'a')
    fh.write(bestparams + '\n' + "解释方差：" + str(explained_variance_score(y_test, predicted_test)) + '\n' + "r2_score:" + str(r2_score(y_test, predicted_test)))
    fh.close()
    del X_test, y_test
    gc.collect()
    print('old')
    '''
    # 使用邻接矩阵回归
    NX, ny = read_new()
    time_start = time.time()
    Nreg = rf_fit_model_k_fold(NX, ny)
    Nclf = ensemble.RandomForestRegressor(random_state=1,n_estimators=Nreg.best_params_['n_estimators'],max_features=Nreg.best_params_['max_features'],max_depth = Nreg.best_params_['max_depth'])
    NX_train, NX_ver, ny_train, ny_ver = train_test_split(NX, ny, test_size=0.2)
    del NX, ny
    gc.collect()
    Nclf.fit(NX_train, ny_train)
    Npredicted = Nclf.predict(NX_ver)
    # 保存最优学习配置及验证集表现
    Nresult = np.vstack((ny_ver, Npredicted))
    Ndataframe = pd.DataFrame({'y_ver': Nresult[0], 'predicted': Nresult[1]})
    Ndataframe.to_csv("rf_new_ver.csv", index=True, sep=',')
    Nbestparams = "n_estimators:" + str(Nreg.best_params_['n_estimators']) + 'max_features:' + str(Nreg.best_params_['max_features']) + "max_depth:" + str(Nreg.best_params_['max_depth'])
    fh = open("rf_new_ver.txt", 'a')
    fh.write(Nbestparams + '\n' + "解释方差：" + str(explained_variance_score(ny_ver, Npredicted)) + '\n' + "r2_score:" + str(r2_score(ny_ver, Npredicted)))
    fh.close()
    del NX_train, NX_ver, ny_train, ny_ver
    gc.collect()
    # 保存最优学习器测试集表现
    NX_test, ny_test = read_new_test()
    Npredicted_test = Nreg.predict(NX_test)
    Nresult_test = np.vstack((ny_test, Npredicted_test))
    Ndataframe_test = pd.DataFrame({'y_test': Nresult_test[0], 'predicted': Nresult_test[1]})
    Ndataframe_test.to_csv("rf_new_test.csv", index=True, sep=',')
    fh = open("rf_new_test.txt", 'a')
    fh.write(Nbestparams + '\n' + "解释方差：" + str(explained_variance_score(ny_test, Npredicted_test)) + '\n' + "r2_score:" + str(r2_score(ny_test, Npredicted_test)))
    fh.close()
    del NX_test, ny_test
    gc.collect()
    time_end = time.time()
    print(time_end - time_start)
    '''
if __name__ == "__main__":
    usemlp()
    usesvm()
    userf()



