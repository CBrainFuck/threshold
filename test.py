# -*- coding: UTF-8 -*-
from __future__ import division
import numpy as np
import networkx as nx
import Queue
import random as rd
import scipy.integrate as spi
from multiprocessing import Process

N = 40                                 #节点个数
NO = 30
#gillespi函数输入节点数，邻接矩阵以及β,γ可以返回在初始条件为1各节点染毒的情况下200条样本路径,每次迭代1000次是否灭绝
def gillespi(N,a,gama,beta):
    Mbeta = np.multiply(beta,a)                                               # 构造染毒速率矩阵
    Mgama =  np.zeros(N) + gama                                               # 构造重装速率矩阵
    rads = np.arange(0, N, 1)
    np.random.shuffle(rads)  # 随机选取初始染毒节点
    Endflag = np.zeros(100)                                               # 100条样本路径中每次迭代1000次的演化结果
    for count in range(0,Endflag.size):
        #Eflag = 1                                                        #迭代终止标识0标识灭绝1标识不灭绝
        x = np.zeros(N)                                                  # 定义节点状态向量
        degree = np.zeros(N)                                              # 存储各节点的度数
        for i in range(0, N):
            degree[i] = np.sum(a[i])
        position = np.argmax(degree)                                      # position为度数最大节点编号
        x[position] = 1                                                   # 将度数最大节点即中心节点作为初始染毒节点
        #x[rads[0]] = 1                                                   # 将随机初始条件中的染毒节点状态更新为1
        # x[0] = 1                                                          # 固定0号节点为初始染毒节点
        for k in range(0,200):
            V = np.zeros(N)                                            # 存储每个节点的转移速率
            Vtemp = np.zeros(N + 1)                                    # 储存循环到某个节点时当前所有节点总速率.第一位为0从第二位开始储存
            #计算每个节点的转移速率
            for i in range(0,N):
                neighbor = np.nonzero(a[i])                            #取出某节点邻接矩阵中的相邻点下标返回类型为元组
                if x[i] == 0:                                         #如果为脆弱节点
                    for j in range(0,neighbor[1].size):               #对邻居节点进行循环
                        if x[neighbor[1][j]] == 1:                    #如果邻居节点为染毒节点则累计向该节点传毒的速率
                            V[i] = V[i] + Mbeta[i,neighbor[1][j]]
                if x[i] ==1:                                          #如果为染毒节点则转移速率为重装速率
                    V[i] = Mgama[i]
                Vtemp[i+1] = Vtemp[i] + V[i]
            if Vtemp[N] == 0:                                         #所有节点的染毒速率均为0表示病毒灭绝迭代终止
                #Eflag = 0
                break
            for i in range(0,N+1):
                Vtemp[i] = Vtemp[i] / Vtemp[N]                        #将当前累加速率/所有节点总速率得到一个0-1的概率值用于选择转移节点
            #选择一个要改变状态的节点并进行状态转移
            Nflag = np.random.rand()
            for i in range(0,N):
                if (Nflag >= Vtemp[i]) and (Nflag < Vtemp[i+1]):      #随机数落在某个区间
                    #print(i)
                    if x[i] == 0:                                     #若为脆弱节点则改为染毒反之亦然
                        x[i] =1
                    else:
                        x[i] = 0
                    break                                            #选择一个节点并转移后跳出循环
            #Endflag[k] = np.sum(x==1) /  N
        #print Endflag
        Endflag[count] = np.sum(x)                                    #保存每条样本路径最后的染毒节点数
        #Endflag[count] =  Eflag
    #print Endflag.size
    if (np.sum(Endflag) / (100*N))< 0.01:
        return 0
    else:
        return 1
#给定邻接矩阵和γ值寻找β阈值
def getbeta(N,gama,a):
    l = np.linalg.eigvals(a)                    #计算邻接矩阵特征值
    l = l.astype(np.float16)
    lowbeta = (1 / (l.max())) * gama     #依据τ构造初始下界
    """
    while 1:                         #β以γ递增直到找到第一个不灭绝的β值
        if gillespi(N,a,gama,highbeta) == 1:
            break
        else:
            highbeta = highbeta + gama
    """
    D = np.zeros((N,N))
    for i in range(0,N):
        D[i][i] = np.sum(a[i])
    Laplacian = D - a                        #构造拉普拉斯矩阵
    f = np.linalg.eigvals(Laplacian)                #计算拉普拉斯矩阵特征值
    f = f.astype(np.float16)
    f = np.sort(f)
    highbeta = (1 / f[1]) * gama             #依据τ构造初始上界
    #print(highbeta)
    #print(lowbeta)
    while highbeta - lowbeta > 0.001:   #二分查找β阈值区间长度小于3位小数时终止
        cbeta = (highbeta + lowbeta) / 2
        if gillespi(N,a,gama,cbeta) == 0:
            lowbeta = cbeta
        else:
            highbeta = cbeta
    return (highbeta + lowbeta) / 2


def ws():                              #  产生不同小世界网络进行数据生成的函数
    d = np.zeros(N)
    for i in range(0,5):
        print(i)                                    #通过控制台观察当前计算到了第几个网络
        for j in range(4,24,1):
            ws = nx.watts_strogatz_graph(N, j, 0.3)  # 生成小世界网络
            a = nx.adjacency_matrix(ws).todense()    # 取出所生成网络的邻接矩阵
            gama = 0                                 #重装速率γ
            for k in range(0,10):
                print("ws")                      #通过控制台观察每个网络进度如何
                gama =gama + 0.1
                b= np.zeros(N)
                b[0] = gama
                b[1]=getbeta(N,gama,a)
                c = np.vstack((a, b))                # 将β,γ放在邻接矩阵下一行空白填充0
                d = np.vstack((d, c))                #将每次迭代的输出矩阵叠加
    d = d[1:, ]
    np.savetxt("C:/Users/Jason Chen/PycharmProjects/test/123/ws.txt", d)


def ba():                             #  产生不同无标度网络进行数据生成的函数
    d = np.zeros(N)
    for i in range(0, 5):
        print(i)  # 通过控制台观察当前计算到了第几个网络
        for j in range(1,21,1):
            ws = nx.random_graphs.barabasi_albert_graph(N, j)
            a = nx.adjacency_matrix(ws).todense()            # 取出所生成网络的邻接矩阵
            gama = 0  # 重装速率γ
            for k in range(0, 10):
                print("ba")  # 通过控制台观察每个网络进度如何
                gama = gama + 0.1
                b = np.zeros(N)
                b[0] = gama
                b[1] = getbeta(N, gama, a)
                c = np.vstack((a, b))  # 将β,γ放在邻接矩阵下一行空白填充0
                d = np.vstack((d, c))  # 将每次迭代的输出矩阵叠加
    d = d[1:, ]
    np.savetxt("C:/Users/Jason Chen/PycharmProjects/test/123/ba.txt", d)

def rn():
    m = np.loadtxt(open("politician_edges.csv"), delimiter=",", usecols=(0), skiprows=1)
    n = np.loadtxt(open("politician_edges.csv"), delimiter=",", usecols=(1), skiprows=1)
    A = np.zeros((5908, 5908))
    for i in range(0, 41729):  # 根据边矩阵构造邻接矩阵，重边算一条边
        A[int(m[i])][int(n[i])] = 1
        A[int(n[i])][int(m[i])] = 1
    for i in range(0, 5908):  # 删除图中的自环
        A[i][i] = 0
    d = np.zeros(N)
    for i in range(0, 100):  # 6000节点的网络产生7000个样本
        print("pol" + str(i))  # 通过控制台观察当前计算到了第几个网络
        #  以下为从真实网络中切出50个节点的网络
        node = np.zeros(1)  # 存储选择出的节点
        node = node.astype('int32')
        start = int((np.random.random() * 5907) + 0.5)  # 随机产生初始节点
        node[0] = start
        choose = start  # 标识当前正在遍历的节点
        flag = 0  # 循环次数标识，防止陷入小于50个节点的连通片
        q = Queue.Queue()  # 队列用于广度优先遍历
        while (node.size < 50):  # 广度优先找到50个节点
            flag = flag + 1
            neighbor = np.nonzero(A[choose])
            for e in neighbor[0]:
                if e not in node:
                    node = np.append(node, e)
            for e in neighbor[0]:
                q.put(e)
            if (q.empty() == 1):  # 若某次队列为空则重置初始条件从头开始
                node = np.zeros(1)  # 存储选择出的节点
                node = node.astype('int32')
                start = int((np.random.random() * 5907) + 0.5)  # 随机产生初始节点
                node[0] = start
                choose = start  # 标识当前正在遍历的节点
                flag = 0  # 循环次数标识，防止陷入小于50个节点的连通片
                q = Queue.Queue()  # 队列用于广度优先遍历
                continue  # 跳出本次循环开始下一次即从头开始
            choose = q.get()
            while (np.random.random() < 0.35):  # 增加样本多样性，每次选出邻居节点有35%概率不要该节点继续向后选择
                if (q.empty() == 1):  # 若某次队列为空则重置初始条件从头开始
                    node = np.zeros(1)  # 存储选择出的节点
                    node = node.astype('int32')
                    start = int((np.random.random() * 5907) + 0.5)  # 随机产生初始节点
                    node[0] = start
                    choose = start  # 标识当前正在遍历的节点
                    flag = 0  # 循环次数标识，防止陷入小于50个节点的连通片
                    q = Queue.Queue()  # 队列用于广度优先遍历
                    break
                choose = q.get()
            if flag > 200:  # 循环200次没得到结果，陷入小于50个节点的连通片，将所有变量还原重新开始
                node = np.zeros(1)  # 存储选择出的节点
                node = node.astype('int32')
                start = int((np.random.random() * 5907) + 0.5)  # 随机产生初始节点
                node[0] = start
                choose = start  # 标识当前正在遍历的节点
                flag = 0  # 循环次数标识，防止陷入小于50个节点的连通片
                q = Queue.Queue()  # 队列用于广度优先遍历
        node = node[0:50:1]
        a = A[node]
        a = a[:, node]
        a = np.mat(a)
        gama = 0
        for j in range(0,10):
            gama = gama + 0.1  # 重装速率γ
            b = np.zeros(N)
            b[0] = gama
            b[1] = getbeta(N, gama, a)
            c = np.vstack((a, b))  # 将β,γ放在邻接矩阵下一行空白填充0
            d = np.vstack((d, c))  # 将每次迭代的输出矩阵叠加
    d = d[1:, ]
    np.savetxt("C:/Users/Jason Chen/PycharmProjects/test/123/rn.txt", d)

def politician():
    m = np.loadtxt(open("politician_edges.csv"), delimiter=",", usecols=(0), skiprows=1)
    n = np.loadtxt(open("politician_edges.csv"), delimiter=",", usecols=(1), skiprows=1)
    A = np.zeros((5908, 5908))
    for i in range(0, 41729):  # 根据边矩阵构造邻接矩阵，重边算一条边
        A[int(m[i])][int(n[i])] = 1
        A[int(n[i])][int(m[i])] = 1
    for i in range(0, 5908):  # 删除图中的自环
        A[i][i] = 0
    d = np.zeros(N)
    for i in range(0, 5000):            #6000节点的网络产生7000个样本
        print("pol" + str(i))  # 通过控制台观察当前计算到了第几个网络
        #  以下为从真实网络中切出50个节点的网络
        node = np.zeros(1)  # 存储选择出的节点
        node = node.astype('int32')
        start = int((np.random.random() * 5907) + 0.5)  # 随机产生初始节点
        node[0] = start
        choose = start  # 标识当前正在遍历的节点
        flag = 0  # 循环次数标识，防止陷入小于50个节点的连通片
        q = Queue.Queue()               #  队列用于广度优先遍历
        while (node.size < N):  # 广度优先找到50个节点
            flag = flag + 1
            neighbor = np.nonzero(A[choose])
            for e in neighbor[0]:
                if e not in node:
                    node = np.append(node, e)
            for e in neighbor[0]:
                q.put(e)
            if(q.empty() == 1):                   #若某次队列为空则重置初始条件从头开始
                node = np.zeros(1)  # 存储选择出的节点
                node = node.astype('int32')
                start = int((np.random.random() * 5907) + 0.5)  # 随机产生初始节点
                node[0] = start
                choose = start  # 标识当前正在遍历的节点
                flag = 0  # 循环次数标识，防止陷入小于50个节点的连通片
                q = Queue.Queue()  # 队列用于广度优先遍历
                continue                        #跳出本次循环开始下一次即从头开始
            choose = q.get()
            while(np.random.random()<0.35):       #增加样本多样性，每次选出邻居节点有35%概率不要该节点继续向后选择
                if (q.empty() == 1):             #若某次队列为空则重置初始条件从头开始
                    node = np.zeros(1)  # 存储选择出的节点
                    node = node.astype('int32')
                    start = int((np.random.random() * 5907) + 0.5)  # 随机产生初始节点
                    node[0] = start
                    choose = start  # 标识当前正在遍历的节点
                    flag = 0  # 循环次数标识，防止陷入小于50个节点的连通片
                    q = Queue.Queue()      #  队列用于广度优先遍历
                    break
                choose = q.get()
            if flag > 200:  # 循环200次没得到结果，陷入小于50个节点的连通片，将所有变量还原重新开始
                node = np.zeros(1)  # 存储选择出的节点
                node = node.astype('int32')
                start = int((np.random.random() * 5907) + 0.5)  # 随机产生初始节点
                node[0] = start
                choose = start  # 标识当前正在遍历的节点
                flag = 0  # 循环次数标识，防止陷入小于50个节点的连通片
                q = Queue.Queue()      #  队列用于广度优先遍历
        node = node[0:N:1]
        a = A[node]
        a = a[:, node]
        a = np.mat(a)
        gama = 0.5  # 重装速率γ
        b = np.zeros(N)
        b[0] = gama
        b[1] = getbeta(N, gama, a)
        c = np.vstack((a, b))  # 将β,γ放在邻接矩阵下一行空白填充0
        d = np.vstack((d, c))  # 将每次迭代的输出矩阵叠加
    d = d[1:, ]
    np.savetxt("40_politician.txt", d)
def tvshow():
    m = np.loadtxt(open("tvshow_edges.csv"), delimiter=",", usecols=(0), skiprows=1)
    n = np.loadtxt(open("tvshow_edges.csv"), delimiter=",", usecols=(1), skiprows=1)
    A = np.zeros((3892, 3892))
    for i in range(0, 17262):  # 根据边矩阵构造邻接矩阵，重边算一条边
        A[int(m[i])][int(n[i])] = 1
        A[int(n[i])][int(m[i])] = 1
    for i in range(0, 3892):  # 删除图中的自环
        A[i][i] = 0
    d = np.zeros(N)
    for i in range(0, 5000):               #4000节点的网络产生5000个样本
        print("tvs" + str(i))  # 通过控制台观察当前计算到了第几个网络
        #  以下为从真实网络中切出50个节点的网络
        node = np.zeros(1)  # 存储选择出的节点
        node = node.astype('int32')
        start = int((np.random.random() * 3891) + 0.5)  # 随机产生初始节点
        node[0] = start
        choose = start  # 标识当前正在遍历的节点
        flag = 0  # 循环次数标识，防止陷入小于50个节点的连通片
        q = Queue.Queue()               #  队列用于广度优先遍历
        while (node.size < 50):  # 广度优先找到50个节点
            flag = flag + 1
            neighbor = np.nonzero(A[choose])
            for e in neighbor[0]:
                if e not in node:
                    node = np.append(node, e)
            for e in neighbor[0]:
                q.put(e)
            if (q.empty() == 1):  # 若某次队列为空则重置初始条件从头开始
                node = np.zeros(1)  # 存储选择出的节点
                node = node.astype('int32')
                start = int((np.random.random() * 3891) + 0.5)  # 随机产生初始节点
                node[0] = start
                choose = start  # 标识当前正在遍历的节点
                flag = 0  # 循环次数标识，防止陷入小于50个节点的连通片
                q = Queue.Queue()  # 队列用于广度优先遍历
                continue  # 跳出本次循环开始下一次即从头开始
            choose = q.get()
            while (np.random.random() < 0.35):  # 增加样本多样性，每次选出邻居节点有35%概率不要该节点继续向后选择
                if (q.empty() == 1):  # 若某次队列为空则重置初始条件从头开始
                    node = np.zeros(1)  # 存储选择出的节点
                    node = node.astype('int32')
                    start = int((np.random.random() * 3891) + 0.5)  # 随机产生初始节点
                    node[0] = start
                    choose = start  # 标识当前正在遍历的节点
                    flag = 0  # 循环次数标识，防止陷入小于50个节点的连通片
                    q = Queue.Queue()  # 队列用于广度优先遍历
                    break
                choose = q.get()
            if flag > 200:  # 循环200次没得到结果，陷入小于50个节点的连通片，将所有变量还原重新开始
                node = np.zeros(1)  # 存储选择出的节点
                node = node.astype('int32')
                start = int((np.random.random() * 3891) + 0.5)  # 随机产生初始节点
                node[0] = start
                choose = start  # 标识当前正在遍历的节点
                flag = 0  # 循环次数标识，防止陷入小于50个节点的连通片
                q = Queue.Queue()      #  队列用于广度优先遍历
        node = node[0:50:1]
        a = A[node]
        a = a[:, node]
        a = np.mat(a)
        gama = 0.5  # 重装速率γ
        b = np.zeros(N)
        b[0] = gama
        b[1] = getbeta(N, gama, a)
        c = np.vstack((a, b))  # 将β,γ放在邻接矩阵下一行空白填充0
        d = np.vstack((d, c))  # 将每次迭代的输出矩阵叠加
    d = d[1:, ]
    np.savetxt("tvshow.txt", d)
def government():
    m = np.loadtxt(open("government_edges.csv"), delimiter=",", usecols=(0), skiprows=1)
    n = np.loadtxt(open("government_edges.csv"), delimiter=",", usecols=(1), skiprows=1)
    A = np.zeros((7057, 7057))
    for i in range(0, 89455):  # 根据边矩阵构造邻接矩阵，重边算一条边
        A[int(m[i])][int(n[i])] = 1
        A[int(n[i])][int(m[i])] = 1
    for i in range(0, 7057):  # 删除图中的自环
        A[i][i] = 0
    d = np.zeros(NO)
    for i in range(0, 5000):        #7000节点的网络产生8000个样本
        print("gov"+ str(i))  # 通过控制台观察当前计算到了第几个网络
        #  以下为从真实网络中切出50个节点的网络
        node = np.zeros(1)  # 存储选择出的节点
        node = node.astype('int32')
        start = int((np.random.random() * 7056) + 0.5)  # 随机产生初始节点
        node[0] = start
        choose = start  # 标识当前正在遍历的节点
        flag = 0  # 循环次数标识，防止陷入小于50个节点的连通片
        q = Queue.Queue()               #  队列用于广度优先遍历
        while (node.size < NO):  # 广度优先找到50个节点
            flag = flag + 1
            neighbor = np.nonzero(A[choose])
            for e in neighbor[0]:
                if e not in node:
                    node = np.append(node, e)
            for e in neighbor[0]:
                q.put(e)
            if(q.empty() == 1):                   #若某次队列为空则重置初始条件从头开始
                node = np.zeros(1)  # 存储选择出的节点
                node = node.astype('int32')
                start = int((np.random.random() * 7056) + 0.5)  # 随机产生初始节点
                node[0] = start
                choose = start  # 标识当前正在遍历的节点
                flag = 0  # 循环次数标识，防止陷入小于50个节点的连通片
                q = Queue.Queue()  # 队列用于广度优先遍历
                continue                        #跳出本次循环开始下一次即从头开始
            choose = q.get()
            while (np.random.random() < 0.35):  # 增加样本多样性，每次选出邻居节点有35%概率不要该节点继续向后选择
                if (q.empty() == 1):  # 若某次队列为空则重置初始条件从头开始
                    node = np.zeros(1)  # 存储选择出的节点
                    node = node.astype('int32')
                    start = int((np.random.random() * 7056) + 0.5)  # 随机产生初始节点
                    node[0] = start
                    choose = start  # 标识当前正在遍历的节点
                    flag = 0  # 循环次数标识，防止陷入小于50个节点的连通片
                    q = Queue.Queue()  # 队列用于广度优先遍历
                    break
                choose = q.get()
            if flag > 200:  # 循环200次没得到结果，陷入小于50个节点的连通片，将所有变量还原重新开始
                node = np.zeros(1)  # 存储选择出的节点
                node = node.astype('int32')
                start = int((np.random.random() * 7056) + 0.5)  # 随机产生初始节点
                node[0] = start
                choose = start  # 标识当前正在遍历的节点
                flag = 0  # 循环次数标识，防止陷入小于50个节点的连通片
                q = Queue.Queue()      #  队列用于广度优先遍历
        node = node[0:NO:1]
        a = A[node]
        a = a[:, node]
        a = np.mat(a)
        gama = 0.5  # 重装速率γ
        b = np.zeros(NO)
        b[0] = gama
        b[1] = getbeta(NO, gama, a)
        c = np.vstack((a, b))  # 将β,γ放在邻接矩阵下一行空白填充0
        d = np.vstack((d, c))  # 将每次迭代的输出矩阵叠加
    d = d[1:, ]
    np.savetxt("30_government.txt", d)

if __name__ == '__main__':
    p1 = Process(target=politician)
    #p2 = Process(target=tvshow)
    p3 = Process(target=government)
    p1.start()
    #p2.start()
    p3.start()
    p1.join()
    #p2.join()
    p3.join()

