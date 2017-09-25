# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 15:51:04 2017

FCM for MPCs

@author: Zhiming Huang
"""
from KPoweMeans import *
class FCM(KPMCluster):
    def __init__(self,dataSet,m):
        KPMCluster.__init__(self,dataSet)
        self.m=2
        self.dataSet=self.dataset
        self.dataset=array([self.dataset[:,0]*self.tvar**0.5/(self.tmx**2), sin(self.dataset[:,1])*cos(self.dataset[:,2])/2, sin(self.dataset[:,1])*sin(self.dataset[:,2])/2, cos(self.dataset[:,1])/2, sin(self.dataset[:,3])*cos(self.dataset[:,4])/2,sin(self.dataset[:,3])*sin(self.dataset[:,4])/2, cos(self.dataset[:,3])/2]).T

    def Singletraining(self):
        self.row, self.col=self.dataset.shape
        cmin=int(self.kmin)
#        cmax=int(self.kmax)
        cmax=10
        self.DBk=list()
        self.CHk=list()
        self.GDk=list()
        self.XBk=list()
        self.PBk=list()
        centroids=list()
        CHopt=0
        self.ClusterAssment=list()
        self.kopt=0
        self.U=list()
        for c in range(cmin,cmax+1):
 #           u=random.random_sample((self.row,c))
 #           for i in range(self.row):
 #               u[i]=u[i]/sum(u[i])
            self.centroids=self.InitialGuess(c,self.distance).A
            self.u, self.centroids=self.alg_Pfcm(self.centroids)
            self.U.append(self.u)
            self.clusterAssment=mat(self.classify())
            self.ClusterAssment.append(self.clusterAssment)
            L=array([size(nonzero(self.clusterAssment[:,0].A==i)[0]) for i in range(c)])
            self.DBk.append(self.DB(c,L))
            self.CHk.append(self.CH(c,L))
            self.GDk.append(self.GD53(k,L))
            self.XBk.append(self.XB(k,L))
            self.PBk.append(self.PBM(k,L))
            centroids.append(self.centroids)  
        self.Kr(cmin,cmax,centroids)
        print(self.kopt)
                           

#cluster accoding to matrix u    
    def classify(self):
        a=zeros([self.row,1])
        for i in range(self.row):
            b=sort(self.u[i])
            if b[-1]-b[-2]>0.2:
                a[i]=self.u[i].tolist().index(max(self.u[i]))
            else:
                a[i]=inf
        return a
        
     
 
    def alg_Pfcm(self, centroids, e=0.0001, terminateturn=500):  
        u1 = zeros([self.row,size(centroids,axis=0)])
        k = 0
        while(True):
            # calculate new u matrix    
            u2 = self.fcm_u(centroids,self.distance)
            # calculate one more turn  
            centroids = self.fcm_c(u2)
            # max difference between u1 and u2  
            maxu = self.fcm_maxu(u1,u2)
            
            if (maxu < e):
                break
            u1 = u2
            k=k+1
            if k > terminateturn:
                break
            print(u2)
        return u2, centroids

    def fcm_oneturn(self,u):   
        # calculate centroids of clusters  

        return u2, centroids

    def fcm_maxu(self,u1,u2):  

        err = 0
        n = self.row
        c = size(u2,axis=1)
        for i in range(n):  
            for j in range(c):  
                err = max(fabs(u1[i][j] - u2[i][j]), err) 
        return err
    
    def normalize(self,a,b):
        c=list()
        for i in range(size(a)):
            c.append(max(a[i],b[i]))
        return array(c)
        
    
    def distance(self,a,b):
        return sqrt(sum(square((a - b))))

    def fcm_u(self,centroids,distance):  

        n = self.row
        c = size(centroids,axis=0)  
        u = zeros([n,c])
        for i in range(n):  
            for j in range(c):
                sum1 = 0
                d1 = distance(self.dataset[i],centroids[j])  
                for k in range(c):
                    d2 = distance(self.dataset[i],centroids[k])
                    if d2!= 0:  
                        sum1 += power(d1/d2, float(2)/(float(self.m)-1))  
                if sum1!=0:  
                    u[i][j] = 1/sum1

        return u
 
    def fcm_c(self,u2):  

        n = self.row
        c = size(u2,axis=1)
        centroids = []
        for i in range(c):
            sum1 = 0;
            sum2 = 0;
            for j in range(n):
                sum2 += self.Powerx[j]*power(u2[j][i], self.m)
                sum1 += self.Powerx[j]*dot(self.dataset[j], power(u2[j][i], self.m))
            if sum2!= 0:
                cj = sum1/sum2
            else:
                cj = zeros([1,self.col])
            centroids.append(cj)

        return array(centroids)
    
    def CH(self, k,L):#Calinski-Harabasz index
        centroids=mat(self.centroids)
        clusterAssment=self.clusterAssment
        avc=sum(multiply(mat(self.dataset),mat(self.Powerx).T),axis=0)/sum(self.Powerx)
        trB=sum(array([L[i]*self.distance(centroids.A[i],avc[0])**2 for i in range(k)]))
        trW=0
        for i in range(k):
            ptsInCluster = self.dataset[nonzero(clusterAssment[:,0].A==i)[0]]
            for j in range(L[i]):
                trW+=self.distance(ptsInCluster[j],centroids.A[i])**2
        return trB*(sum(L)-k)/(trW*(k-1))
 
    def DB(self,k, L,t=2):
        centroids=mat(self.centroids)
        clusterAssment=self.clusterAssment
        #Davis-Bouldin index
        R=zeros([k,1])
        for i in range(k):
            ptsInCluster = self.dataset[nonzero(clusterAssment[:,0].A==i)[0]]
            ski=sum(array([self.distance(ptsInCluster[j], centroids.A[i]) for j in range(L[i])]))/L[i]
            R[i]=0
            for j in range(k):
                if j==i:
                    continue
                dij=self.distance(centroids.A[i],centroids.A[j])
                ptsInCluster = self.dataset[nonzero(clusterAssment[:,0].A==j)[0]]
                skj=sum(array([self.distance(ptsInCluster[lk], centroids.A[j]) for lk in range(L[j])]))/L[i]
                if R[i] < (ski+skj)/dij:
                    R[i]=(ski+skj)/dij
        return sum(R)/k
    
    def GD53(self,k,L):
        centroids=mat(self.centroids)
        clusterAssment=self.clusterAssment          
        a=0
        b=0
        for k1 in range(k):
            ptsInCluster1 = self.dataset[nonzero(clusterAssment[:,0].A==k1)[0]]
            for k2 in range(k):
                if k1==k2:
                    continue
                ptsInCluster2 = self.dataset[nonzero(clusterAssment[:,0].A==k2)[0]]
                delta=sum(array([self.distance(ptsInCluster[lk], centroids.A[k1]) for lk in range(L[k1])]))+sum(array([self.MCD(ptsInCluster[lk], centroids.A[k2]) for lk in range(L[k2])]))
                a=min(delta/(L[k1]+L[k2]), a)
        for i in range(k):
            delta=2*sum(array([self.distance(ptsInCluster[lk], centroids.A[i]) for lk in range(L[i])]))
            b=max(delta,b)
        return a/b
    
    def XB(self,k,L):
        centroids=mat(self.centroids)
        clusterAssment=self.clusterAssment         
        a=0
        b=0
        for i in range(k):
            a+=sum(array([self.distance(ptsInCluster[lk], centroids.A[i])**2 for lk in range(L[i])]))
        for k1 in range(k):
            for k2 in range(k):
                if k1==k2:
                    continue
                dis=self.distance(centroids.A[k1],centroids.A[k2])
                b=min(dis,b)
        return a/(sum(L)*b**2)

#PNB index                
    def PBM(self,k,L):
        centroids=mat(self.centroids)
        clusterAssment=self.clusterAssment         
        a=0
        b=0
        for k1 in range(k):
            for k2 in range(k):
                if k1==k2:
                    continue
                dis=self.distance(centroids.A[k1],centroids.A[k2])
                b=max(dis,b)    
        for i in range(k):
            a+=sum(array([self.distance(ptsInCluster[lk], centroids.A[i]) for lk in range(L[i])]))
        return (b/(k*a))**2    