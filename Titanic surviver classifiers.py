
# coding: utf-8

# In[1]:


# notes
# to chec the unmber of unique values for a coloumn we use .value_counts()
# we alwas use base estimator t get  and set parameters for grid serach and piplines and also to select specific attibutes
# scikit-learn:Operates on numpy matrices, not DataFrames
# two very important classes has been created ; columns selector class and categorical imputer (DataFrameSelector)(MostFrequentImputer)
# most frequect imputer is a class to fill missing values with the most frequent values in a coloumn 


# In[2]:


# we need to uild a classifier to see whether a specpfpc man was survived or not based on his attributes 
# S1: GET THE DATA
import pandas as pd 
train = pd.read_csv('train.csv')
train.info()# below ou can see age is missing some values and cabin also but cabin is not so impoartant and we can ignore


# In[3]:


# tae a look on the numbrs of he dataframe
train.describe()
train['Sex'].unique()


# In[4]:


# let's see how many people survived 
train['Survived'].value_counts() #342 had been survived 


# In[5]:


# what about the class
print(train["Pclass"].value_counts())
# what about the sex
print(train["Sex"].value_counts())


# In[6]:


# we have to make a coloum selector from pandas as sicket does not deal with pandas
from sklearn.base import BaseEstimator, TransformerMixin
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, selected_clmns):
        self.selected_clmns = selected_clmns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.selected_clmns]


# In[7]:


# let's creta a pipleline for two steps (selec coloumns and mputer for missed  numeric values of the selected columns)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer as SimpleImputer
# num_pipeline = Pipeline([
#         ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
#         ("imputer", SimpleImputer(strategy="median")),
#     ])
numeric_values_pipe= Pipeline([('numiric_ coloumns' ,DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
                               ('imputer_for_missed' ,SimpleImputer(strategy='median'))])


# In[38]:


num =numeric_values_pipe.fit_transform(train)# see below we selected only 4 num coloumns and get the missed values  based on median


# In[9]:


# let's  select a categorical data create an imputer for thier missed data 
#we build a calss to fill missed cat data with most frequent 
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


# In[36]:


# we will appl a pipe line for three steps (select columns , fill with missed data and then convert to numerical)
from sklearn.preprocessing import OneHotEncoder   
cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
        ("imputer", MostFrequentImputer())])
ndf = cat_pipeline.fit_transform(train)


# In[46]:


from sklearn.preprocessing import LabelEncoder

from sklearn import preprocessing
def hot_encoder(cat_features):
    enc = preprocessing.LabelEncoder()
    enc.fit(cat_features)
    new_cat_features = enc.transform(cat_features)
    new_cat_features = new_cat_features.reshape(-1, 1) # Needs to be the correct shape
    ohe = preprocessing.OneHotEncoder(sparse=False) #Easier to read
    nope = ohe.fit_transform(new_cat_features)
    return(nope)
array2D_1 = hot_encoder(ndf['Pclass'])
array2D_2 = hot_encoder(ndf['Sex'])
array2D_3 = hot_encoder(ndf['Embarked'])
import numpy as np
cat = np.concatenate((array2D_1,array2D_2,array2D_3),axis=1)
xtrain = np.concatenate((num , cat),axis = 1)
xtrain.shape


# In[47]:


# our label
ytrain = train['Survived']
ytrain.shape


# In[51]:


# Lat's try with svm classifier
from sklearn.svm import SVC
svm_clf = SVC(gamma="auto")
svm_clf.fit(xtrain , ytrain)


# In[53]:


# let's process our  test data 
test = pd.read_csv('test.csv')
# test.head()
num_test =numeric_values_pipe.fit_transform(test)
num_test


# In[55]:


cat_test =cat_pipeline.fit_transform(test) 
cat_test


# In[60]:


pclass = hot_encoder(cat_test['Pclass'])
sex = hot_encoder(cat_test['Sex'])
embarf = hot_encoder(cat_test['Embarked'])
cat_testf = np.concatenate((pclass,sex,embarf),axis=1)
xtest = np.concatenate((num_test , cat_testf),axis = 1)
xtest.shape


# In[72]:


from sklearn.model_selection import cross_val_score

svm_scores = cross_val_score(svm_clf, xtrain, ytrain, cv=10)
svm_scores.mean()
print(svm_scores)


# In[73]:


# Let's try a RandomForestClassifier:
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_scores = cross_val_score(forest_clf, xtrain, ytrain, cv=10)
forest_scores.mean()#better ccuracy than SVC
print(forest_scores )


# In[71]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.plot([1]*10, svm_scores, ".")
plt.plot([2]*10, forest_scores, ".")
plt.boxplot([svm_scores, forest_scores], labels=("SVM","Random Forest"))
plt.ylabel("Accuracy", fontsize=14)
plt.show()

