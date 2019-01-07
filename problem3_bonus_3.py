import scipy.io as sio
from sklearn.cluster import KMeans
import numpy as np
import random
import matplotlib.pyplot as plt


def plot(matr,C,k):
    colors = []
    for i in range(k):
        colors.append((random.randint(0, 255)/255.0,
            random.randint(0, 255)/255.0,
            random.randint(0, 255)/255.0))
    for idx,value in enumerate(C):
        plt.plot(matr[idx][0],matr[idx][1],'o',color=colors[int(C[idx])])

    plt.show()


if __name__ == '__main__':
    cluster_num = 2
    k = 5
    file = 'ringData.mat'
    data1 = sio.loadmat(file)
    data = np.array(data1['Dataset'])
    points_num = len(data)
    dis_matr = np.zeros((points_num,points_num))
    W = np.zeros((points_num,points_num))
    for i in range(points_num):
        for j in range(i+1,points_num):
            dis_matr[i][j] = dis_matr[j][i] = np.linalg.norm(data[i]-data[j])
    for idx,each in enumerate(dis_matr):
        index_array  = np.argsort(each)
        W[idx][index_array[1:k+1]] = 1
    tmp_W = np.transpose(W)
    W = (tmp_W+W)/2

    points_num = len(W)
    D = np.diag(np.zeros(points_num))
    for i in range(points_num):
        D[i][i] = sum(W[i])
    L = D-W
    e2,e1 = np.linalg.eig(L)
    dim = len(e2)
    dicte2 = dict(zip(e2,range(0,dim)))
    kEig = np.sort(e2)[0:cluster_num]
    ix = [dicte2[k] for k in kEig]
    e2 = e2[ix]
    e1 = e1[:, ix]


    clf = KMeans(n_clusters=cluster_num)
    print(clf)
    s = clf.fit(e1)
    print(s)
    C = s.labels_
    centers1 = []
    for i in range(max(C)+1):
        points_list = np.where(C==i)[0].tolist()
        centers1.append(np.average(data[points_list],axis=0))

    plot(data,s.labels_,cluster_num)


    label = KMeans(n_clusters=2).fit_predict(data)

    plot(data, label, cluster_num)


