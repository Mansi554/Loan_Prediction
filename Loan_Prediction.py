#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df= pd.read_csv('train.csv')


# In[7]:


df.head()


# In[9]:


df.describe()


# In[11]:


df.info()


# In[13]:


df.isnull().sum()


# In[16]:


df['LoanAmount']=df['LoanAmount'].fillna(df['LoanAmount'].mean())
df['Loan_Amount_Term']=df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean())
df['Credit_History']=df['Credit_History'].fillna(df['Credit_History'].mean())


# In[19]:


df['Gender']=df['Gender'].fillna(df['Gender'].mode()[0])
df['Married']=df['Married'].fillna(df['Married'].mode()[0])
df['Dependents']=df['Dependents'].fillna(df['Dependents'].mode()[0])
df['Self_Employed']=df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])


# In[48]:


df.isnull().sum()


# In[21]:


import seaborn as sns


# In[22]:


sns.countplot(df['Education'])


# In[24]:


sns.countplot(df['Self_Employed'])


# In[26]:


sns.countplot(df['Property_Area'])


# In[32]:


sns.countplot(df['Loan_Status'])


# In[30]:


sns.displot(df['ApplicantIncome'])


# In[33]:


df['ApplicantIncome']=np.log(df['ApplicantIncome'])


# In[34]:


sns.displot(df['ApplicantIncome'])


# In[43]:


df['LoanAmount']=np.log(df['LoanAmount'])


# In[45]:


sns.displot(df["LoanAmount"])


# In[47]:


sns.displot(df['Loan_Amount_Term'])


# In[49]:


sns.displot(df['Credit_History'])


# In[51]:


df['Total_Income']=df['ApplicantIncome']+ df['CoapplicantIncome']
df.head()


# In[59]:


corr= df.corr()
sns.heatmap(corr, annot= True , cmap="BuPu")


# In[95]:


from sklearn.preprocessing import LabelEncoder
cols = ["Gender","Married","Education","Self_Employed","Property_Area","Loan_Status",'Dependents']
le= LabelEncoder()
for col in cols:
       df[col]=le.fit_transform(df[col])


# In[96]:


df.head()


# In[ ]:





# In[98]:


df.head()


# In[107]:


df= df.drop(['Loan_ID'], axis = 'columns')


# In[108]:


df.head()


# In[109]:


x= df.drop(columns=['Loan_Status'],axis =1)
y= df['Loan_Status']


# In[110]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test= train_test_split(x,y,test_size=0.25,random_state=42)


# In[111]:


x_train.shape


# In[112]:


from sklearn.model_selection import cross_val_score
def classify(model,x,y):
    x_train, x_test, y_train,y_test= train_test_split(x,y,test_size=0.25,random_state=42)
    model.fit(x_train,y_train)
    print("Accuracy is", model.score(x_test,y_test)*100)
    
    score= cross_val_score(model,x,y,cv=5)
    print("cross validation is ",np.mean(score)*100)


# In[113]:


from sklearn.linear_model import LogisticRegression
model= LogisticRegression()
classify(model,x,y)


# In[114]:


model= LogisticRegression()
model.fit(x_train,y_train)


# In[121]:


y_pred= model.predict(x_test)


# In[125]:


from sklearn.metrics import confusion_matrix
confusion_matrix= confusion_matrix(y_test,y_pred)
print(confusion_matrix)


# In[126]:


sns.heatmap(confusion_matrix, annot = True)


# In[ ]:




