#  -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 10:53:29 2017
KpowerMeans
@author: Zhiming
"""

#coding=utf-8
from numpy import *

#数据转换成向量 /delay/EoA/AoA/EoD/AoD/Power/

class KPMCluster:
      def __init__(self, dataSet): 
          self.dataset=dataSet[:,0:-1]
          self.kmin=2
          self.kmax=size(dataSet,axis=0)/2.0
          self.tmx=max(dataSet[:,0])-min(dataSet[:,0])
          self.Powerx=dataSet[:,-1]
          self.tvar=var(self.dataset[:,0])*0.5
          
      def training(self,ite):
          self.ite=ite
          kmin=int(self.kmin)
#          kmax=int(self.kmax)
          kmax=10
          self.DBk=list()
          self.CHk=list()
          self.GDk=list()
          self.XBk=list()
          self.PBk=list()
          centroids=list()
          self.ClusterAssment=list()
          self.kopt=0
          for k in range(kmin, kmax+1):
              self.centroids, self.clusterAssment=self.kPowerMeans(k,distMeas=self.MCD, createCent=self.InitialGuess)
              self.ClusterAssment.append(self.clusterAssment)
              L=array([size(nonzero(self.clusterAssment[:,0].A==i)[0]) for i in range(k)])
              self.DBk.append(self.DB(k,L))
              self.CHk.append(self.CH(k,L))
              self.GDk.append(self.GD53(k,L))
              self.XBk.append(self.XB(k,L))
              self.PBk.append(self.PBM(k,L))
              centroids.append(self.centroids)
          self.Kr(kmin,kmax,centroids)
          self.dataset, self.Powerx, self.clusterAssment=self.Prune()
          
      def CHDB(self, kmin, kmax, centroids):   
          CHopt=0
          for i in range(kmin,kmax+1):
              if self.DBk[i-kmin]<=2*min(self.DBk):
                  if self.CHk[i-kmin]>CHopt:
                      CHopt=self.CHk[i-kmin]
                      self.kopt=i
                      self.centroids=array(centroids[i-kmin])
                      self.clusterAssment=array(self.ClusterAssment[i-kmin])
 
      def Kr(self, kmin, kmax, centroids):          
          rank=zeros([1,kmax+1-kmin])
          for i in range(kmin,kmax+1):
              rank[i-kmin]+=sort(self.CHk).tolist().index(self.CHk[i-kmin])
              rank[i-kmin]+=sort(self.GDK).tolist().index(self.GDK[i-kmin])
              rank[i-kmin]-=sort(self.XBk).tolist().index(self.XBk[i-kmin])
              rank[i-kmin]+=sort(self.PBk).tolist().index(self.PBk[i-kmin])
          Hrank=sort(rank)
          self.kopt=rank.tolist().index(Hrank[-1])
          self.centroids=array(centroids[self.kopt-kmin])
          self.clusterAssment=array(self.ClusterAssment[self.kopt-kmin])
     
          
#计算两个向量的距离，用的是MCD
      def MCD(self,vA,vB):
          mcd1=(0.5*sum(array([[sin(vA[1])*cos(vA[2])-sin(vB[1])*cos(vB[2])],[sin(vA[1])*cos(vA[2])-sin(vB[1])*cos(vB[2])],[cos(vA[1])-cos(vB[1])]])**2))**0.5
          mcd2=(0.5*sum(array([[sin(vA[3])*cos(vA[4])-sin(vB[3])*cos(vB[4])],[sin(vA[3])*cos(vA[4])-sin(vB[3])*cos(vB[4])],[cos(vA[3])-cos(vB[4])]])**2))**0.5
          mcd3=(vA[0]-vB[0])*self.tvar**0.5/(self.tmx**2)
          return sum(mcd1**2+mcd2**2+mcd3**2)**0.5
          


#随机生成初始的质心   
      def randCent(self, k):
          n = shape(self.dataset)[1]
          centroids = mat(zeros((k,n)))
          for j in range(n):
              minJ = min(self.dataset[:,j])
              rangeJ = float(max(array(self.dataset)[:,j]) - minJ)
              centroids[:,j] = minJ + rangeJ * random.rand(k,1)
          return centroids
      
      def InitialGuess(self,k,dist):
          n=shape(self.dataset)[1]
          centroids = mat(zeros((k,n)))
          l=[self.Powerx.tolist().index(max(self.Powerx))]
          centroids[0]=self.dataset[l[0]]
          d=list()
          for i in range(2,n):
              for j in range(len(self.dataset)):
                  if j not in l :
                      d.append(array([min([dist(self.dataset[j], centroids[k]) for k in range(i-1)]),j]))
              index=array(d)[:,0].tolist().index(max(array(d)[:,0]))
              l.append(d[index][-1])
              centroids[i]=self.dataset[l[-1]]
          return centroids
              
                  
        
        
      def kPowerMeans(self, k, distMeas, createCent):
          m = shape(self.dataset)[0]
          clusterAssment = mat(zeros((m,2)))#create mat to assign data points 
          centroids = createCent(k)
          clusterChanged = True
#          for iteration in range(self.ite):
          iteration=0
          while clusterChanged:
              iteration += 1
              if iteration>self.ite:
                  break
              clusterChanged = False
              for i in range(m):#for each data point assign it to the closest centroid
                  minDist = inf
                  minIndex = -1
                  for j in range(k):
                      distJI = self.Powerx[i]*distMeas(centroids.A[j,:],self.dataset[i,:])
                      if distJI < minDist:
                          minDist = distJI; minIndex = j
                          if clusterAssment[i,0] != minIndex:
                             clusterChanged = True
                  clusterAssment[i,:] = int(minIndex),minDist**2
                  print (centroids)
              for cent in range(k):#recalculate centroids
                  ptsInCluster = self.dataset[nonzero(clusterAssment[:,0].A==cent)[0]]#get all the point in this cluster .A 转换成array类
                  Powery=self.Powerx[nonzero(clusterAssment[:,0].A==cent)[0]]
                  if size(ptsInCluster)==0:
                      centroids[cent,:]=centroids[cent,:]
                  else:
                      centroids[cent,:] = sum(multiply(mat(ptsInCluster),mat(Powery).T),axis=0)/sum(Powery)
          return centroids, clusterAssment

      def CH(self, k,L):#Calinski-Harabasz index
          centroids=self.centroids
          clusterAssment=self.clusterAssment
          avc=sum(multiply(mat(self.dataset),mat(self.Powerx).T),axis=0)/sum(self.Powerx)
          trB=sum(array([L[i]*self.MCD(centroids.A[i],avc.A[0])**2 for i in range(k)]))
          trW=0
          for i in range(k):
              ptsInCluster = self.dataset[nonzero(clusterAssment[:,0].A==i)[0]]
              for j in range(L[i]):
                  trW+=self.MCD(ptsInCluster[j],centroids.A[i])**2
          return trB*(sum(L)-k)/(trW*(k-1))
 
      def DB(self,k, L,t=2):
          centroids=self.centroids
          clusterAssment=self.clusterAssment
          #Davis-Bouldin index
          R=zeros([k,1])
          for i in range(k):
              ptsInCluster = self.dataset[nonzero(clusterAssment[:,0].A==i)[0]]
              ski=sum(array([self.MCD(ptsInCluster[j], centroids.A[i]) for j in range(L[i])]))/L[i]
              R[i]=0
              for j in range(k):
                  if j==i:
                    continue
                  dij=self.MCD(centroids.A[i],centroids.A[j])
                  ptsInCluster = self.dataset[nonzero(clusterAssment[:,0].A==j)[0]]
                  skj=sum(array([self.MCD(ptsInCluster[lk], centroids.A[j]) for lk in range(L[j])]))/L[i]
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
                  delta=sum(array([self.MCD(ptsInCluster[lk], centroids.A[k1]) for lk in range(L[k1])]))+sum(array([self.MCD(ptsInCluster[lk], centroids.A[k2]) for lk in range(L[k2])]))
                  a=min(delta/(L[k1]+L[k2]), a)
          for i in range(k):
              delta=2*sum(array([self.MCD(ptsInCluster[lk], centroids.A[i]) for lk in range(L[i])]))
              b=max(delta,b)
          return a/b
       
      def XB(self,k,L):
          centroids=mat(self.centroids)
          clusterAssment=self.clusterAssment         
          a=0
          b=0
          for i in range(k):
              a+=sum(array([self.MCD(ptsInCluster[lk], centroids.A[i])**2 for lk in range(L[i])]))
          for k1 in range(k):
              for k2 in range(k):
                  if k1==k2:
                      continue
                  dis=self.MCD(centroids.A[k1],centroids.A[k2])
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
                  dis=self.MCD(centroids.A[k1],centroids.A[k2])
                  b=max(dis,b)    
          for i in range(k):
              a+=sum(array([self.MCD(ptsInCluster[lk], centroids.A[i]) for lk in range(L[i])]))
          return (b/(k*a))**2
#      def compare(self,vector1,vector2):
#          for i in range(size(vector1,axis=0)):
#              for j in range(size(vector2[i],axis=0)):
#                  if vector1[i][j]<=vector2[i][j]:
#                      return False
#          return True
      def compare(self,x,y):
          for i in range(size(x,axis=0)):
              for j in range(size(x[i],axis=0)):
                  if x[i][j]<= y[i][j]:
                      return False
          return True
          

        
      def Prune(self,s=0.9,p=0.9):
          Powerk=list()
          Sk=list()
          dataset=self.dataset
          clusterAssment=self.clusterAssment
          dataSet=hstack((self.dataset,self.clusterAssment))
          powerx=self.Powerx
          for i in range(self.kopt):
              Powerk.append(self.Powerx[nonzero(mat(self.clusterAssment)[:,0].A==i)[0]])
              Sk.append(self.dataset[nonzero(mat(self.clusterAssment)[:,0].A==i)[0]])
          Power0=Powerk
          Sk0=Sk
          for i in range(self.kopt):
              Dist0=0
              while sum(Powerk[i])>0.9*sum(Power0[i]) and self.compare(Sk[i],0.9*Sk0[i]):
                  for k in range(size(Sk,axis=0)):
                      D=self.MCD(Sk[k],self.Centroids[i])
                      if D>Dist0:
                          Dist0=D
                          index=k
                  index2=nonzero(mat(self.clusterAssment)[:,0].A==i)[0][index]
                  dataSet=delete(dataSet,index2,0)
                  powerx=delete(powerx,index2,0)
                  clusterAssment=dataSet[:,-2:]
                  dataset=dataSet[:,:-2]
                  Powerk[i]=powerx[mat(nonzero(clusterAssment)[:,0].A==i)[0]]
                  Sk[i]=dataset[mat(nonzero(clusterAssment)[:,0].A==i)[0]]
          return dataset, powerx, clusterAssment


      def showCluster(self):
          numSamples, dim = self.dataset.shape
          from mpl_toolkits.mplot3d.axes3d import Axes3D
          from matplotlib import pyplot as plt
          import matplotlib as mpl
          mpl.style.use('default')
          fig=plt.figure()
          ax=Axes3D(fig)
          mark = ['v', '^', '<', '>', '1', '2', '3', '4', '8', 's'] 
          color= ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C0']
          for i in range(numSamples):
              markIndex = int(self.clusterAssment[i, 0])  
              ax.scatter(self.dataset[i, 3], self.dataset[i, 1], self.dataset[i, 0], c=color[markIndex],S=20)  
          for i in range(self.kopt):  
              ax.scatter(self.centroids[i, 3], self.centroids[i, 1], self.centroids[i, 0], c=color[i],marker=mark[i],s=50)
          return fig
    
    