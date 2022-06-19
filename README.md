# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the necessary packages.
2. Read the given csv file and display the few contents of the data.
3. Import K Means and use for loop to calculate the within cluster sum of squares the data.
4. Plot the wcss for each iteration, also known as the elbow method plot.
5. Predict the clusters and plot them. 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Rajalakshmi R
RegisterNumber: 212219040116 
*/
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("Mall_Customers.csv")
df.head()

df.info()
df.isnull().sum()

from sklearn.cluster import KMeans
wcss = []  
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = "k-means++")
    kmeans.fit(df.iloc[:,3:])
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")
plt.show()
km = KMeans(n_clusters = 5)
km.fit(df.iloc[:,3:])
y_pred = km.predict(df.iloc[:,3:])
df["cluster"] = y_pred
df0 = df[df["cluster"]==0]
df1 = df[df["cluster"]==1]
df2 = df[df["cluster"]==2]
df3 = df[df["cluster"]==3]
df4 = df[df["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="yellow",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="pink",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="cyan",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="skyblue",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="violet",label="cluster4")
plt.legend()
plt.title("Customer Segments")
plt.show()
```

## Output:
### Initial Dataset:
![Screenshot (50)](https://user-images.githubusercontent.com/87656716/174478677-94a981cf-8f74-4139-ac12-009d9081dcce.png)
### Data Information:
![Screenshot (52)](https://user-images.githubusercontent.com/87656716/174478755-57d54d44-d34e-47dc-858b-2e3bb488cb83.png)
### Elbow graph:
![Screenshot (54)](https://user-images.githubusercontent.com/87656716/174478818-4805df12-4d70-421a-a96f-a4044654d617.png)
### Cluster Graph:
![Screenshot (56)](https://user-images.githubusercontent.com/87656716/174478876-7b88e7d4-6226-40f2-882d-e0bb31b199b7.png)




## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
