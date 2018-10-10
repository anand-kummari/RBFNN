from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import math

X=[]
high_dimension=6
Sigma=[]
output=[]
label=[]
f=open("stock_input2.txt","r")
for line in f:
        X.append(list(map(float,line.strip().split("\t"))))
# print(X)
f.close()
# X = np.array([[1, 2], [1, 4], [1, 0],
#                [4, 2], [4, 4], [4, 0]])
kmeans = KMeans(n_clusters=6, random_state=0).fit(X)
# kmeans.labels_
# array([0, 0, 0, 1, 1, 1], dtype=int32)
# kmeans.predict([[0, 0], [4, 4]])
# array([0, 1], dtype=int32)
# print("centers: ",kmeans.cluster_centers_)
# array([[ 1.,  2.],
#        [ 4.,  2.]])
dist_sum=0
for i in range(0,high_dimension):
    for j in range(0,high_dimension):
        dist=euclidean_distances(kmeans.cluster_centers_,kmeans.cluster_centers_)
        # print("before:  ",dist[i][j])
        dist[i][j]=math.pow(dist[i][j],2)
        # print("after: ",dist[i][j])
        dist_sum=dist_sum+dist[i][j]
    Sigma.append(math.sqrt(dist_sum/high_dimension))
# print(Sigma)
Y = kmeans.cluster_centers_.T
# print(Y.shape)
# Y=np.reshape(kmeans.cluster_centers_)
f=open("testfile.txt","r")
for line in f:
        # label.append(list(map(float,line.strip())))
        label.append(float(line.strip()))
# label=np.asarray(label)
# print(label)
f.close()
X_train=[]
X_test=[]
y_train=[]
y_test=[]

X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.30)
print("X_train: ",len(X_train))
print("y_train: ",y_train)
new_X = rbf_kernel(X_train,Y)
print("new X:   ",new_X)
# print(len(new_X))
SLP=Perceptron(alpha=0.0001,max_iter=10,eta0=1.0)#need to add y from aman
SLP.fit(new_X,label)
output=SLP.predict(new_X)
print(output.shape)
accuracy=accuracy_score(output,label)
print(accuracy)