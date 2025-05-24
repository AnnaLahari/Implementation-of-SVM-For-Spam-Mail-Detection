#!/usr/bin/env python
# coding: utf-8

# In[2]:


print("A.LAHARI")
print("212223230111")
import chardet
file='spam.csv'
with open(file,'rb')as rawdata:
    result=chardet.detect(rawdata.read(100000))
result


# In[3]:


import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')
data.head()


# In[4]:


data.info()


# In[5]:


data.isnull().sum()


# In[6]:


x=data["v2"].values
y=data["v1"].values


# In[7]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[10]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)


# In[11]:


from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred


# In[12]:


from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

