# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM :
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required :
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm :
1. Import pandas module and import the required data set.
2. Find the null values and count them.
3. Count number of left values.
4. From sklearn import LabelEncoder to convert string values to numerical values.
5. From sklearn.model_selection import train_test_split.
6. Assign the train dataset and test dataset.
7. From sklearn.tree import DecisionTreeClassifier.
8. Use criteria as entropy.
9. From sklearn import metrics.
10. Find the accuracy of our model and predict the require values.

## Program & Output :
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Narendran K
RegisterNumber: 212223230135
*/
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
```

![image](https://github.com/user-attachments/assets/0dd52c9e-4216-4024-bbda-887d88f6cdf5)

```
data.info()
```

![image](https://github.com/user-attachments/assets/bab9a236-6f0d-4c95-ac1c-b64b9275800a)

```
data.isnull().sum()
```

![image](https://github.com/user-attachments/assets/3ad5ef0c-8fcf-4e03-a2e6-3d776332c90a)

```
data["left"].value_counts()
```

![image](https://github.com/user-attachments/assets/754e13ef-6ac7-4ee9-90f9-9e1cee383026)

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
```

![image](https://github.com/user-attachments/assets/31ea068f-9cf9-488e-963b-f6abaa34638d)

```
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","Work_accident","promotion_last_5years","salary"]]
x.head()
```

![image](https://github.com/user-attachments/assets/2d1eb539-2fd4-4bae-b007-9156d7e60791)

```
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

![image](https://github.com/user-attachments/assets/580cf1fe-306b-4ff2-92fa-8863f1ed93de)

```
dt.predict([[0.5,0.8,9,260,6,0,1,]])
```

![image](https://github.com/user-attachments/assets/435bcfcb-1077-40df-81ff-ddf0bd29ea68)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
