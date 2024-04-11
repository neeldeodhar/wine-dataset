#!/usr/bin/env python
# coding: utf-8

# In[82]:


#pip install numpy
#pip install pandas
#pip install sklearn
#pip install ucimlrepo


# In[83]:


# importing libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import tree
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[84]:


#reading dataset

data = pd.read_csv("1712194016263_winequality-red.csv", sep = ';')


# In[85]:


#reading first 5 columns of the dataset.
data.head()


# In[86]:


#getting shape of dataset (rows and columns)
data.shape


# In[87]:


df = pd.DataFrame(data)


# In[88]:


#dropping quality column
y = data.quality
X = data.drop('quality', axis=1)


# In[89]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)


# In[90]:


print(X_train.head())


# In[91]:


X_train_scaled = preprocessing.scale(X_train)
print (X_train_scaled)


# In[92]:


#decision tree classifier trained by the author.
clf=tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)


# In[93]:


confidence = clf.score(X_test, y_test)
print("\nThe confidence score:\n")
print(confidence)


# In[94]:


y_pred = clf.predict(X_test)


# In[95]:


#converting the numpy array to list
x=np.array(y_pred).tolist()

#printing first 5 predictions
print("\nThe prediction:\n")
for i in range(0,5):
    print (x[i])
    
#printing first five expectations
print("\nThe expectation:\n")
print (y_test.head)


# In[96]:


#training Random Forest classifier for testing
score = []
for i in range(1,30):
    model = RandomForestClassifier(random_state = 0, criterion = "entropy" , n_estimators = i)
    model = model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    
    score.append(accuracy_score(y_test, y_predict))


# In[97]:


score


# In[98]:


y_predict


# In[99]:


#plotting accuracy percentage vs n_estimators (random forest)


plt.plot(range(1,30), score)
plt.title("Accuracy percent vs: n_estimators value ")
plt.xlabel("Number of Estimators")
plt.ylabel("Accuracy Rate")

plt.show()


# In[100]:


max(score)


# In[101]:


#training KNeighborsClassifier for testing

from sklearn.neighbors import KNeighborsClassifier 
    
    
scores = []
for k in range (1,30):

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
  
    scores.append(accuracy_score(y_test, y_pred))

print(str(scores))
print ((y_pred))


# In[102]:


#plotting accuracy percentage for KNeighbors Classifiers


plt.plot(range(1,30), scores)
plt.title("Accuracy percent vs: KNeighbors Classifiers ")
plt.xlabel("Number of neighbors")
plt.ylabel("Accuracy Rate")

plt.show()


# In[103]:


max (scores)


# In[104]:


#comparing Random Forest Scores with KNeighbors classifier score


plt.plot(range(1,30),score, scores)
plt.title("Accuracy percent: Random Forest vs: KNeighbors Classifiers ")
plt.xlabel("Number of estimators/ neighbors")
plt.ylabel("Accuracy Rate")

plt.show()


# In[105]:


# conclusion:
print ("The author has tried Decision tree models; and I have tried the Random Forest and KNNeighbors")
print ("based on above analysis, all 3 models offer similar predictions")
print ("However, the Random Forest offers highest accuracy followed by Decision Tree and KNearestNeighbors")

