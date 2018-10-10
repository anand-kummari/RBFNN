import sys
import math
import re
# import matplotlib.pyplot as plt

List = []
colors=[]

f= open("stock_input2.txt","r")
for line in f:
	# List.append((int(line.split(' ')[0]),int(line.split(' ')[1])))
	# List.append((re.split("\t|\n",line)))
	List.append(list(map(float,line.strip().split("\t"))))
# print(List)
f.close()
# f2=open("colors.txt","r")
# for line in f2:
# 	colors.append(line.strip())
# #print colors
# f2.close()
# print "Enter the number of clusters"
# k = input()
# #print num
k=7
clusters = [[] for i in range(0,k)]
centroids=[List[i] for i in range(0,k)]
print("The initial cluster centers are :")
print(centroids)

def distance(a,b):
	x1,x2,x3,x4,x5,x6=a
	y1,y2,y3,y4,y5,y6=b
	return math.sqrt((y1-x1)**2+(y2-x2)**2+(y3-x3)**2+(y4-x4)**2+(y5-x5)**2+(y6-x6)**2)

def min_dis_for(point):
	print("centroids:",centroids[0])
	print("point:",point)
	min_dis=distance(centroids[0],point)
	cluster_number=0
	for i in range(1,k): 
		new_dis=distance(centroids[i],point)
		if new_dis<min_dis:
			min_dis=new_dis
			cluster_number=i
	return cluster_number

def new_centroids():
	new_centroids=[(0,0,0,0,0,0) for i in range(0,k)]
	for i in range(0,k):
		cluster=clusters[i]
		print("cluster ",cluster)
		X,Y=zip(*cluster)
		sum_x,sum_y=sum(X),sum(Y)
		new_centroids[i]=float(sum_x)/len(X),float(sum_y)/len(Y)
	return new_centroids, not new_centroids==centroids

centroids_changed=True
number_of_loops=0

while(centroids_changed):
	number_of_loops=number_of_loops+1
	clusters = [[] for i in range(0,k)]
	for point in List:
		print(point)
		head=min_dis_for(point)
		clusters[head].append(point)
	centroids,centroids_changed=new_centroids()
print("The cluster centroids at the end : ")
print(centroids)
# print "The points in all clusters : "
# print clusters
# print "the number of loops are :"
# print number_of_loops
# j=0
# for cluster in clusters:
# 	x,y=zip(*cluster)
# 	plt.scatter(x,y,label="dots",color=colors[j],marker=".",s=100)
# 	j+=1
# for centroid in centroids:
# 	plt.scatter(centroid[0],centroid[1],color="black",marker=".",s=100)
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.title("Clusters")

# plt.show()



