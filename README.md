# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries (pandas, chardet, sklearn, etc.).

2.Detect the encoding of the CSV file using chardet.

3.Read the CSV file with the correct encoding.

4.Check the data for structure and missing values.

5.Split the data into input (x = messages) and output (y = labels).

6.Divide the data into training and testing sets.

7.Convert text data into numbers using CountVectorizer.

8.Train an SVM model, make predictions, and calculate accuracy.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: A.LAHARI
RegisterNumber:  212223230111
*/
print("A.LAHARI")
print("212223230111")
import chardet
file='spam.csv'
with open(file,'rb')as rawdata:
    result=chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')
data.head()

data.info()

data.isnull().sum()

x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:

![image](https://github.com/user-attachments/assets/62185625-c8d4-4c2d-b40e-c44befc909ac)

## data.head()

![image](https://github.com/user-attachments/assets/8222e966-7fe2-40fd-8c0d-bc2a8e01061b)

## data.info()

![image](https://github.com/user-attachments/assets/a0ad64fb-5d2b-466b-a43c-ced13ac919f3)

## data.isnull().sum()

![image](https://github.com/user-attachments/assets/f393921d-6820-413a-979d-b2e436fab8fc)

## y_pred

![image](https://github.com/user-attachments/assets/b56c91f7-1611-42ab-add3-b1da7260dc73)

## accuracy()

![image](https://github.com/user-attachments/assets/3974113f-2b2b-42a8-b90c-90000035f06a)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
