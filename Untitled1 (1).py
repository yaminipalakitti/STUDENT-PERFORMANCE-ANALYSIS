#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import csv


# In[20]:


df = pd.read_csv(r"C:\Users\SASIDHAR\Downloads\StudentsPerformance.csv")


# In[21]:


df


# In[23]:


categorical_cols = df.select_dtypes(include='object').columns


# In[24]:


categorical_cols


# In[25]:


for i in categorical_cols:
    print(df[i].unique())


# In[26]:


df.isnull().sum()


# In[27]:


import seaborn as sns


# In[28]:


sns.countplot(df['gender'])


# In[29]:


count_test = df['test preparation course'].value_counts()
labels = df['test preparation course'].value_counts().index
plt.figure(figsize= (6,6))
plt.pie(count_test,labels=labels,autopct='%1.1f%%')
plt.legend(labels)
plt.show()


# In[30]:


df['average_score']=(df['math score']+df['reading score']+df['writing score'])/3


# In[31]:


df


# In[32]:


sns.scatterplot(x=df['average_score'],y=df['math score'],hue=df['gender'])


# In[33]:


sns.scatterplot(x=df['average_score'],y=df['reading score'],hue=df['gender'])


# In[34]:


df


# In[35]:


gender = {
    'male':1,
    'female':0
}


# In[36]:


race = {
    'group A':0,
    'group B':1,
    'group C':2,
    'group D':3,
    'group E':4
}


# In[37]:


df['gender']=df['gender'].map(gender)
df['race/ethnicity']=df['race/ethnicity'].map(race)


# In[38]:


df


# In[39]:


level = {
    "bachelor's degree":0,
    'some college':1,
    "master's degree":2,
    "associate's degree":3,
    "high school":4,
    "some high school":5
}


# In[40]:


df['parental level of education']=df['parental level of education'].map(level)


# In[41]:


df


# In[42]:


df = pd.get_dummies(df,drop_first=True)


# In[43]:


df


# In[44]:


x = df.drop(columns='average_score').values


# In[45]:


x


# In[46]:


x[0]


# In[47]:


y = df['average_score'].values


# In[48]:


y


# In[49]:


from sklearn.model_selection import train_test_split


# In[50]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[51]:


from sklearn.ensemble import RandomForestRegressor


# In[52]:


model=RandomForestRegressor()


# In[53]:


model.fit(x_train,y_train)


# In[54]:


predictions=model.predict(x_test)


# In[55]:


predictions


# In[56]:


from sklearn.metrics import r2_score


# In[57]:


print(r2_score(predictions,y_test))

