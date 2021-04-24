#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm


# In[2]:


def data_process(fname):
    data=pd.read_csv(fname)
    train=data.sample(frac=0.7) 
    test=data.drop(train.index)
    return np.array(train.iloc[:,:-1].T),np.array(train.iloc[:,-1].T),np.array(test.iloc[:,:-1].T),np.array(test.iloc[:,-1].T)


# In[15]:


class LogisticRegression:
    def __init__(self,nof):
        self.W = np.random.uniform(low=0.0, high=1.0, size=nof)
        self.b = 0.
        self.nof=nof
    def sigmoid(self,z):
        s = 1. / (1 + np.exp(-z))
        return s
    def E(self,x,y):
        z=x.T@self.W+self.b
        ycap=self.sigmoid(z)
        e1=0
        e2=0
        for i in range(x.shape[1]):
            if(ycap[i]!=0):
                e1-=y[i]*np.log(ycap[i])
            if(ycap[i]!=1):
                e2-=(1-y[i])*np.log(1-ycap[i])
            
        e=(e1+e2)
        return e
    def fit_sgd(self, X, y, epochs,lr,eps,freq):
        N = X.shape[1]
        error=[]
        acc=[]
        epr=0
        i=0
        e=0
        a=0
        acc.append(0)
        for n in range(epochs):
            for i in range(N):
                z=X.T@self.W+self.b
                ycap = self.sigmoid(z)
                self.W+= lr * X[:,i]*(y[i]-ycap[i])
                self.b+=lr*(y[i]-ycap[i])
                if(ycap[i]==0):
                    e=-1*((1-y[i])*np.log(1-ycap[i]))
                elif(ycap[i]==1):
                    e=-1*(y[i]*np.log(ycap[i]))
                else:
                    e=-1*((y[i]*np.log(ycap[i]))+(1-y[i])*np.log(1-ycap[i]))
                a=(1-np.mean(np.abs(ycap-y)))*100
            if(n%freq==0):
                acc.append(a)
                error.append(e)
            
        return error,acc,self.W
        
    def fit_gd(self, X, y, epochs,lr,eps,freq): 
        error=[]
        acc=[]
        epr=0
        a=0
        for n in range(epochs):
            z=X.T@self.W+self.b
            ycap = self.sigmoid(z)
            self.W -= lr *(X@(ycap-y))
            self.b-=lr*np.sum(ycap-y)
            e=self.E(X, y)
            a=(1-np.mean(np.abs(ycap-y)))*100
            if(n%freq==0):
                error.append(e)
                acc.append(a)
        return error,acc,self.W
            
    def predict(self,x_test):
        z = x_test.T @ self.W+self.b
        y=self.sigmoid(z)
        y[y >=0.5] = 1
        y[y <0.5] = 0
        return y


# In[4]:


def evaluate(Y_train,Y_prediction_train,Y_test,Y_prediction_test):
    ic=np.abs(Y_prediction_train - Y_train)
    tp=len(np.where(Y_prediction_train + Y_train==2)[0])
    tn=len(np.where(Y_prediction_train + Y_train==0)[0])
    fp=len(np.where(Y_prediction_train - Y_train==-1)[0])
    fn=len(np.where(Y_prediction_train - Y_train==1)[0])
    
    tp1=len(np.where(Y_prediction_test + Y_test==2)[0])
    tn1=len(np.where(Y_prediction_test + Y_test==0)[0])
    fp1=len(np.where(Y_prediction_test - Y_test==-1)[0])
    fn1=len(np.where(Y_prediction_test - Y_test==1)[0])
    
    a=(tp+tn)/(tp+tn+fp+fn)
    p=tp/(tp+fp)
    r=tp/(tp+fn)
    f=2*p*r/(p+r)
    
    a1=(tp1+tn1)/(tp1+tn1+fp1+fn1)
    p1=tp1/(tp1+fp1)
    r1=tp1/(tp1+fn1)
    f1=2*p1*r1/(p1+r1)
    return a,a1,f,f1,r,r1,p,p1


# In[16]:


fname="/Users/durbasatpathi/Documents/ml/a2/dataset_LR.csv"
df=pd.read_csv(fname)
nof=df.shape[1]-1
lr=[0.001,0.05,0.1]
emain1=[]
acmain1=[]
for l in lr:
    print(l)
    train_acc=[]
    test_acc=[]
    error=[]
    acc=[]
    train_f=[]
    test_f=[]
    train_p=[]
    test_p=[]
    train_r=[]
    test_r=[]

    for i in range(10):
        print("Split "+ str(i+1) +": ")
        X_train,y_train,X_test,y_test=data_process("/Users/durbasatpathi/Documents/ml/a2/dataset_LR.csv")
        model=LogisticRegression(nof)
        e,a,w=model.fit_sgd(X_train,y_train,1000,l,0.00001,50)
        error.append(e)
        y_pred_tst=model.predict(X_test)
        y_pred_tr=model.predict(X_train)
        tr_acc,tst_acc,tr_f,tst_f,tr_r,tst_r,tr_p,tst_p=evaluate(y_train,y_pred_tr,y_test,y_pred_tst)
        print("w",w)
        train_acc.append(tr_acc)
        test_acc.append(tst_acc)
        train_f.append(tr_f)
        test_f.append(tst_f)
        train_p.append(tr_p)
        test_p.append(tst_p)
        train_r.append(tr_r)
        test_r.append(tst_r)
        if(i==9):
            emain1.append(e)
            acmain1.append(a)

    print(f"Overall Training Accuracy = {np.mean(train_acc)}")
    print(f"Overall Test Accuracy = {np.mean(test_acc)}")
    print(f"Overall Training f = {np.mean(train_f)}")
    print(f"Overall Test f = {np.mean(test_f)}")
    print(f"Overall Training p = {np.mean(train_p)}")
    print(f"Overall Test p = {np.mean(test_p)}")
    print(f"Overall Training r = {np.mean(train_r)}")
    print(f"Overall Test r = {np.mean(test_r)}")


# In[17]:


import matplotlib.pyplot as plt
plt.figure(figsize = (10,10))

y=np.array(emain1[0])
plt.plot(y)
y=np.array(emain1[1])
plt.plot(y)
y=np.array(emain1[2])
plt.plot(y)
plt.legend(["0.001","0.05","0.1"], loc ="upper right")
plt.show()


# In[18]:


import matplotlib.pyplot as plt
plt.figure(figsize = (10,10))
for i in range(3):
    y=np.array(acmain1[i]).squeeze()
    plt.plot(y)
plt.legend(["0.001","0.05","0.1"], loc ="upper right")
plt.show()


# In[12]:


fname="/Users/durbasatpathi/Documents/ml/a2/dataset_LR.csv"
df=pd.read_csv(fname)
nof=df.shape[1]-1
lr=[0.001,0.05,0.1]
emain=[]
acmain=[]
for l in lr:
    print(l)
    train_acc=[]
    test_acc=[]
    error=[]
    acc=[]
    train_f=[]
    test_f=[]
    train_p=[]
    test_p=[]
    train_r=[]
    test_r=[]
    for i in range(10):
        print("Split "+ str(i+1) +": ")
        X_train,y_train,X_test,y_test=data_process("/Users/durbasatpathi/Documents/ml/a2/dataset_LR.csv")
        model=LogisticRegression(nof)
        e,a,w=model.fit_gd(X_train,y_train,1000,l,0.00001,1)
        if(i==5):
            emain.append(e)
            acmain.append(a)
        y_pred_tst=model.predict(X_test)
        y_pred_tr=model.predict(X_train)
        tr_acc,tst_acc,tr_f,tst_f,tr_r,tst_r,tr_p,tst_p=evaluate(y_train,y_pred_tr,y_test,y_pred_tst)
        print("w",w)
        train_acc.append(tr_acc)
        test_acc.append(tst_acc)
        train_f.append(tr_f)
        test_f.append(tst_f)
        train_p.append(tr_p)
        test_p.append(tst_p)
        train_r.append(tr_r)
        test_r.append(tst_r)
    print(f"Overall Training Accuracy = {np.mean(train_acc)}")
    print(f"Overall Test Accuracy = {np.mean(test_acc)}")
    print(f"Overall Training f = {np.mean(train_f)}")
    print(f"Overall Test f = {np.mean(test_f)}")
    print(f"Overall Training p = {np.mean(train_p)}")
    print(f"Overall Test p = {np.mean(test_p)}")
    print(f"Overall Training r = {np.mean(train_r)}")
    print(f"Overall Test r = {np.mean(test_r)}")


# In[13]:


import matplotlib.pyplot as plt
plt.figure(figsize = (10,10))
for i in range(3):
    y=np.array(emain[i]).squeeze()
    plt.plot(y)
plt.legend(["0.001","0.05","0.1"], loc ="upper right")
plt.show()


# In[14]:


import matplotlib.pyplot as plt
plt.figure(figsize = (10,10))
for i in range(3):
    y=np.array(acmain[i]).squeeze()
    plt.plot(y)
plt.legend(["0.001","0.05","0.1"], loc ="upper right")
plt.show()


# In[ ]:




