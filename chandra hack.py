#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
train=pd.read_excel('C:/Users/chandu prakash/Downloads/Data_Train.xlsx')
test=pd.read_excel('C:/Users/chandu prakash/Downloads/Data_Test.xlsx')


# In[2]:


np.unique(train.Delivery_Time,return_counts=True)


# In[2]:


Location=pd.get_dummies(train.Location,drop_first=True)


# In[3]:


Location1=pd.get_dummies(test.Location,drop_first=True)


# In[4]:


train["Average_Cost"]= train["Average_Cost"].replace('for','s200')


# In[5]:


import re
import string
def clean_text_round2(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub('%s' % re.escape(string.punctuation),'',text)
    text=re.sub(r'\(.*\)', '',text)
    
    text=re.sub(':%s' % re.escape(string.punctuation),'',text)
    text=re.sub("\D", "", text)
    
    return text
round2=lambda x: clean_text_round2(x)
train['Average_Cost']=train.Average_Cost.apply(round2)
test['Average_Cost']=test.Average_Cost.apply(round2)
train['Minimum_Order']=train.Minimum_Order.apply(round2)
test['Minimum_Order']=test.Minimum_Order.apply(round2)


# In[6]:


train["Minimum_Order"]=train.Minimum_Order.astype(int)
test["Minimum_Order"]=test.Minimum_Order.astype(int)
train["Average_Cost"]=train.Average_Cost.astype(int)
test["Average_Cost"]=test.Average_Cost.astype(int)


# In[7]:


train["Rating"]=train["Rating"].replace('-',3.7)
train["Rating"]=train["Rating"].replace('NEW',3.5)
train["Rating"]=train["Rating"].replace('Opening Soon',0)
train["Rating"]=train["Rating"].replace('Temporarily Closed',1.0)
test["Rating"]=test["Rating"].replace('-',3.7)
test["Rating"]=test["Rating"].replace('NEW',3.5)
test["Rating"]=test["Rating"].replace('Opening Soon',0)


# In[8]:


train["Rating"]=train.Rating.astype(float)


# In[9]:


test["Rating"]=test.Rating.astype(float)


# In[10]:


train["Votes"]= train["Votes"].replace('-',198)
train["Reviews"]=train["Reviews"].replace('-',97)


# In[11]:


test["Votes"]= test["Votes"].replace('-',182)
test["Reviews"]=test["Reviews"].replace('-',87)


# In[12]:


train["Votes"]=train.Votes.astype(int)
train["Reviews"]=train.Reviews.astype(int)


# In[13]:


train['Cuisines']= train['Cuisines'].str.split().str.len()
train.head()
test['Cuisines']= test['Cuisines'].str.split().str.len()
test.head()


# In[14]:


test["Votes"]=test.Votes.astype(int)
test["Reviews"]=test.Reviews.astype(int)


# In[15]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Delivery_Time= le.fit_transform(train.Delivery_Time)


# In[173]:


final=pd.DataFrame(train,columns=['Cuisines','Average_Cost','Votes','Rating'])


# In[174]:


finall=pd.concat([final,Location],axis=1,join='outer')
finall.head(2)
finall.isnull().sum()


# In[175]:


X=finall
y=pd.DataFrame(train,columns=['Delivery_Time'])
y.head(2)


# In[176]:


# splitting the data into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = 0.3,random_state=0)


# In[177]:


from imblearn.over_sampling import SMOTE
import numpy as np
seed=100
k=1
sm = SMOTE(sampling_strategy='auto', k_neighbors=k, random_state=seed)
X_res, y_res= sm.fit_resample(X_train, y_train)


# In[178]:


from sklearn.ensemble import RandomForestClassifier

rnd_clf=RandomForestClassifier(n_estimators=400,max_leaf_nodes=16,n_jobs=-1)
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
bag_clf=BaggingClassifier(DecisionTreeClassifier(),n_estimators=100,max_samples=100,
                         bootstrap=True,n_jobs=-1)
import xgboost
model1=xgboost.XGBClassifier()


# In[179]:


bag_clf.fit(X_res, y_res)
rnd_clf.fit(X_res, y_res)
model1.fit(X_res, y_res)


# In[180]:


y_pred1= bag_clf.predict(X_test)
y_pred2= rnd_clf.predict(X_test)
y_pred3= model1.predict(X_test.values)


# In[181]:


from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

results = confusion_matrix(y_test, y_pred3)
results1 = confusion_matrix(y_test, y_pred2) 
print('Confusion Matrix :')
print(results) 
print('Accuracy Score :',accuracy_score(y_test, y_pred3)) 
print('Report : ')
print(classification_report(y_test, y_pred3)) 
print(classification_report(y_test, y_pred2)) 


# In[182]:


from sklearn.metrics import accuracy_score
print ("Accuracy : ", accuracy_score(y_test, y_pred1)) 
print ("Accuracy : ", accuracy_score(y_test, y_pred2)) 


# In[183]:


final1=pd.DataFrame(test,columns=['Cuisines','Average_Cost','Votes','Rating'])


# In[184]:


mix=pd.concat([final1,Location1],axis=1,join='outer')
mix.head(2)
mixt=mix.values


# In[185]:


predictions=bag_clf.predict(mix)


# In[186]:


result=pd.DataFrame(predictions,columns=["Delivery_Time"])
result.head()
np.unique(result.Delivery_Time,return_counts=True)


# In[187]:


predictions1=model1.predict(mixt)


# In[188]:


result1=pd.DataFrame(predictions1,columns=["Delivery_Time"])
result1.head()
np.unique(result1.Delivery_Time,return_counts=True)


# In[172]:


result.to_excel("kaggle17Nov788.xlsx")


# In[189]:


result1.to_excel("Kaggle1893.xlsx")


# In[ ]:




