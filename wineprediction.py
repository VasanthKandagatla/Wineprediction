#!/usr/bin/env python
# coding: utf-8

# # **Wine prediction project by Group 3**

# In[1]:


import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score


# ## Loading the data set into the project.

# In[2]:


winedata = pd.read_csv("winequality-red.csv")
winedata.head()


# In[3]:


winedata.columns


# ## Checking the correlation for each of the fields

# In[4]:


winedata.corr


# ## Let's do some plotting to know how the data columns are distributed in the dataset

# ### Bivariate analysis/Graphs

# In[5]:


f = plt.figure(figsize = (10,6))


# ### Quality vs fixed acidity

# In[6]:


sns.barplot(x = 'quality', y = 'fixed acidity', ci=None, data = winedata)


# ### Quality vs Volatile acidity

# In[7]:


sns.barplot(x = 'quality', y = 'volatile acidity', ci=None, data = winedata)


# ### Quality vs Alcohol

# In[8]:


sns.barplot(x = 'quality', y = 'alcohol', ci=None, data = winedata)


# ### Quality vs Citric acid

# In[9]:


sns.barplot(x = 'quality', y = 'citric acid', ci=None, data = winedata)


# ### Quality vs Residual Sugar

# In[10]:


sns.barplot(x = 'quality', y = 'residual sugar', ci=None, data = winedata)


# ### Quality vs Chlorides

# In[11]:


sns.barplot(x = 'quality', y = 'chlorides', ci=None, data = winedata)


# ### Quality vs Free Sulfur Dioxide 

# In[12]:


sns.barplot(x = 'quality', y = 'free sulfur dioxide', ci=None, data = winedata)


# ### Quality vs Sulphates

# In[13]:


sns.barplot(x = 'quality', y = 'sulphates',ci=None, data = winedata)


# ### Quality vs Total Sulfur Dioxide

# In[14]:


sns.barplot(x = 'quality', y = 'total sulfur dioxide',ci=None, data = winedata)


# ## Checking correlation between attributes using a heat map

# In[15]:


f, ax = plt.subplots(figsize=(8, 6))
corr = winedata.corr()
sns.heatmap(corr, cmap=sns.diverging_palette(210, 10, as_cmap=True),
            square=True, ax=ax)


# From the above correlation plot for the given dataset for wine quality prediction, we can easily see which items are related strongly with each other and which items are related weekly with each other.

# ### The strongly correlated items are :

# 1.fixed acidity and citric acid. 2.free sulfur dioxide and total sulfur dioxide. 3.fixed acidity and density. 4. alcohol and quality.
# 

# ### From the above holistic picture of heatmap, it is clearly evident that Alcohol is the most important characteristic of any wine taken

# ### The weak correlated items are :

# 1.citric acid and volatile acidity. 2.fixed acidity and ph. 3.density and alcohol.

# In[16]:


sns.pairplot(winedata)


# ### Understanding the data and data pre-processing

# In[17]:


winedata.info()


# In[18]:


winedata.shape


# In[19]:


winedata.describe()


# In[20]:


winedata['quality'].value_counts()


# Removing Unnecassary columns from the dataset<br>
# As we saw that <b>volatile acidity, total sulphor dioxide, chlorides, density</b> are very less related to the dependent variable<br>
# quality so even if we remove these columns the accuracy won't be affected that much.

# In[21]:


#winedata = winedata.drop(['volatile acidity', 'total sulfur dioxide', 'chlorides', 'density'], axis = 1)


# In[22]:


# checking the shape of the dataset
winedata.shape


# In[23]:


winedata.describe()


# In[24]:


winedata['quality'] = winedata['quality'].map({3 : 'bad', 4 :'bad', 5: 'bad',
                                      6: 'good', 7: 'good', 8: 'good'})

# analyzing the different values present in the dependent variable(quality column)
winedata['quality'].value_counts()


# In[25]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

winedata['quality'] = le.fit_transform(winedata['quality'])

winedata['quality'].value_counts


# In[26]:


sns.countplot(winedata['quality'])


# In[27]:


# dividing the dataset into dependent and independent variables

x = winedata.iloc[:,:11]
y = winedata.iloc[:,11]

# determining the shape of x and y.
print(x.shape)
print(y.shape)


# In[94]:


# dividing the dataset in training and testing set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.20,random_state=1)

# determining the shapes of training and testing sets
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[95]:


# standard scaling 
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)


# ### Logistic Regression

# In[96]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# creating the model
model = LogisticRegression()

# feeding the training set into the model
model.fit(x_train, y_train)

# predicting the results for the test set
y_pred = model.predict(x_test)

# calculating the training and testing accuracies
print("Training accuracy :", model.score(x_train, y_train))
print("Testing accuracy :", model.score(x_test, y_test))

# classification report
print(classification_report(y_test, y_pred))

# confusion matrix
print(confusion_matrix(y_test, y_pred))


# ### Support Vector Machine

# In[97]:


from sklearn.svm import SVC

# creating the model
model = SVC()

# feeding the training set into the model
model.fit(x_train, y_train)

# predicting the results for the test set
y_pred = model.predict(x_test)

# calculating the training and testing accuracies
print("Training accuracy :", model.score(x_train, y_train))
print("Testing accuracy :", model.score(x_test, y_test))

# classification report
print(classification_report(y_test, y_pred))

# confusion matrix
print(confusion_matrix(y_test, y_pred))


# ### Decision Tree

# In[98]:


from sklearn.tree import DecisionTreeClassifier

# creating the model
model = DecisionTreeClassifier()

# feeding the training set into the model
model.fit(x_train, y_train)

# predicting the results for the test set
y_pred = model.predict(x_test)

# calculating the training and testing accuracies
print("Training accuracy :", model.score(x_train, y_train))
print("Testing accuracy :", model.score(x_test, y_test))

# classification report
print(classification_report(y_test, y_pred))

# confusion matrix
print(confusion_matrix(y_test, y_pred))


# ### Random Forest

# In[99]:


from sklearn.ensemble import RandomForestClassifier

# creating the model
model = RandomForestClassifier()

# feeding the training set into the model
model.fit(x_train, y_train)

# predicting the results for the test set
y_pred = model.predict(x_test)

# calculating the training and testing accuracies
print("Training accuracy :", model.score(x_train, y_train))
print("Testing accuracy :", model.score(x_test, y_test))

# classification report
print(classification_report(y_test, y_pred))

# confusion matrix
print(confusion_matrix(y_test, y_pred))


# ### Naive Bayes

# In[100]:


from sklearn.naive_bayes import GaussianNB

# creating the model
model = GaussianNB()

# feeding the training set into the model
model.fit(x_train, y_train)

# predicting the results for the test set
y_pred = model.predict(x_test)

# calculating the training and testing accuracies
print("Training accuracy :", model.score(x_train, y_train))
print("Testing accuracy :", model.score(x_test, y_test))

# classification report
print(classification_report(y_test, y_pred))

# confusion matrix
print(confusion_matrix(y_test, y_pred))


# ### Multi Layer Perceptron

# In[101]:


from sklearn.neural_network import MLPClassifier

# creating the model
model = MLPClassifier(hidden_layer_sizes = (100, 100), max_iter = 150)

# feeding the training data to the model
model.fit(x_train, y_train)

# calculating the accuracies
print("training accuracy :", model.score(x_train, y_train))
print("testing accuracy :", model.score(x_test, y_test))

# classification report
print(classification_report(y_test, y_pred))

# confusion matrix
print(confusion_matrix(y_test, y_pred))


# ### Stochastic Gradient Descent Classifier

# In[102]:


from sklearn.linear_model import SGDClassifier

# creating the model
model = SGDClassifier(penalty=None)

# feeding the training data to the model
model.fit(x_train, y_train)

# calculating the accuracies
print("training accuracy :", model.score(x_train, y_train))
print("testing accuracy :", model.score(x_test, y_test))

# classification report
print(classification_report(y_test, y_pred))

# confusion matrix
print(confusion_matrix(y_test, y_pred))


# In[ ]:




