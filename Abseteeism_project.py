#!/usr/bin/env python
# coding: utf-8

# Import the Relevant libraries

# In[21]:


import pandas as pd 
import numpy as np


# In[22]:


##Load the csv data


# In[23]:


df = pd.read_csv('Absenteeism_data.csv')
df.head(10)


# Since ID column doeas not add any weight to our data ,we can remove it

# In[24]:


df1 = df.drop(['ID'],axis=1)
df1


# In[25]:


df1['Education'].unique()


# In[26]:


df1['Education'].value_counts()


# In[27]:


#We can group into two groups by lumping group 2,3 and 4 together.Group 1 are high school leavers and 
# the other group have at tertiary qualification
df1['Education'].map({1:0,2:1,3:1,4:1})


# Dealing with reasons for absence

# In[28]:


#This column can be grouped to few groups as shown on the notes from data.lets display the series.
pd.options.display.max_rows=700
df1['Reason for Absence']


# In[29]:


df1['Reason for Absence'].unique()


# In[30]:


df_reasons_col= pd.get_dummies(df1['Reason for Absence'],drop_first=True)
df_reasons_col


# In[31]:


df_reasons_col.loc[:,1:14].max(axis=1)


# In[32]:


reason_type_1 = df_reasons_col.loc[:, 1:14].max(axis=1)
reason_type_2 = df_reasons_col.loc[:, 15:17].max(axis=1)
reason_type_3 = df_reasons_col.loc[:, 18:21].max(axis=1)
reason_type_4 = df_reasons_col.loc[:, 22:].max(axis=1)


# In[33]:


df2=pd.concat([df1, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis = 1)
df2.head(20)


# In[34]:


df2.columns.values


# In[35]:


cols =['Reason for Absence', 'Date', 'Transportation Expense',
       'Distance to Work', 'Age', 'Daily Work Load Average',
       'Body Mass Index', 'Education', 'Children', 'Pets',
       'Absenteeism Time in Hours','Reason_1','Reason_2','Reason_3','Reason_4']


# In[36]:



df2


# In[37]:


df2.drop(['Reason for Absence'],axis=1)


# In[38]:


df2.columns.values 


# In[39]:


cols


# In[40]:



df2.columns = cols


# In[41]:


df2


# In[42]:


cols_reodered=['Reason_1',
 'Reason_2',
 'Reason_3',
 'Reason_4','Reason for Absence',
 'Date',
 'Transportation Expense',
 'Distance to Work',
 'Age',
 'Daily Work Load Average',
 'Body Mass Index',
 'Education',
 'Children',
 'Pets',
 'Absenteeism Time in Hours'
 ]


# In[43]:


df5 =df2[cols_reodered]


# In[44]:


df5.head(5)


# Dealing with Date

# In[45]:


#we convert date to pandas format
df5['Date']=pd.to_datetime(df5['Date'],format = '%d/%m/%Y')
df5


# In[46]:


#Extracting month from date series
months =[]
for i in range(df5.shape[0]):
    months.append(df5['Date'][i].month)


# In[47]:


months


# In[48]:


df5['Month'] =months
df5


# In[49]:


#Extracting the day of the week
df5['Date'][699].weekday()


# In[50]:


def date_weekday(date_value):
    return date_value.weekday()


# In[51]:


df5['Day of the Week'] = df5['Date'].apply(date_weekday)
df5.head(5)


# In[52]:


df5.columns.values


# In[53]:


cols_updated = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Month','Day of the Week',
       'Reason for Absence', 'Date', 'Transportation Expense',
       'Distance to Work', 'Age', 'Daily Work Load Average',
       'Body Mass Index', 'Education', 'Children', 'Pets',
       'Absenteeism Time in Hours']


# In[54]:


df6 = df5[cols_updated]
df6.head()


# In[55]:


df7=df6.drop(['Date'],axis=1)
df7


# In[56]:


#confirmimg that other variable are numerical
type(df6['Body Mass Index'][0])


# Final Checkpoint

# In[57]:


prep_data=df7.copy()


# In[58]:


prep_data


# In[59]:


#We will apply logistic Regression to this problem.Therefore we must have two outputs.
prep_data['Absenteeism Time in Hours'].median()


# In[60]:


#If time in hours one is absent is more than 3 hours we give it 1 else otherwise 0
targets = np.where(prep_data['Absenteeism Time in Hours']>prep_data['Absenteeism Time in Hours'].median(),1,0)
targets


# In[61]:


prep_data['targets']= targets


# In[62]:


prep_data.head()


# In[63]:


#Lets confirm if our dataset is balanced for ML
#since its 45% and 55%,its fairly balanced
x=(targets.sum()/targets.shape[0])
x


# In[64]:


data_with_targets = prep_data.drop(['Absenteeism Time in Hours'],axis =1)
data_with_targets


# In[69]:


unscaled_inputs = data_with_targets.iloc[:,:-1]
unscaled_inputs


# In[70]:


from sklearn.preprocessing import StandardScaler


# In[80]:


absenteeism_scaler =StandardScaler()
scaled_inputs = scaler.fit_transform(unscaled_inputs)


# Custom_scaler

# In[ ]:





# In[81]:


unscaled_inputs.columns.values


# In[82]:


columns_to_scale = [ 'Month', 'Day of the Week', 'Reason for Absence', 'Transportation Expense',
        'Distance to Work', 'Age', 'Daily Work Load Average',
       'Body Mass Index', 'Children', 'Pets']


# In[83]:


#absenteeism_scaler = CustomScaler(columns_to_scale)


# In[84]:


absenteeism_scaler.fit(unscaled_inputs)


# In[85]:


scaled_inputs =absenteeism_scaler.transform(unscaled_inputs)


# In[86]:


scaled_inputs


# # Splitting and Shuffling of the data

# In[ ]:





# In[87]:


from sklearn.model_selection import train_test_split


# In[88]:


train_test_split(scaled_inputs,targets)


# In[89]:


x_train,x_test,y_train,y_test = train_test_split(scaled_inputs,targets,train_size=0.8,random_state=20)


# In[90]:


print(x_train.shape,y_test.shape)


# Fitting the Model

# In[91]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# In[92]:


log_Reg = LogisticRegression()
log_Reg.fit(x_train,y_train)


# In[93]:


log_Reg.score(x_train,y_train)


# Finding the intercept and Weights

# In[94]:


log_Reg.intercept_


# In[95]:


log_Reg.coef_


# In[96]:


feature_name = unscaled_inputs.columns.values
feature_name


# In[97]:


summary_table = pd.DataFrame(columns=['Feature name'],data = feature_name)
summary_table['Coefficients'] = np.transpose(log_Reg.coef_)
summary_table


# In[98]:


summary_table.index = summary_table.index + 1
summary_table.loc[0]=['intercept',log_Reg.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table


# In[99]:


summary_table['Odds_ratio'] =np.exp(summary_table.Coefficients)
summary_table


# In[100]:


#The odds ratio show how weighty a variable to the problem.For exapmle daily work average and distance to work almost 
#have zero weight and can be comfortably removed without affecting the results.
summary_table.sort_values('Odds_ratio',ascending=False)


# Exploring custom scaler

# In[101]:


#the leading cause of absenteeism is the first four reasons for absence,the first is poisoning,its a serious medical problem
#which must be attended to hence employee will have to seek permission to be absent.


# In[102]:


#Backward elimination
#we can eliminate the variables with less effect
#we can remove daily work load ,distance to work,day of the week


# Testing of the data with Test Data

# In[103]:


#By definition test accuracy is lower than train accuracy
log_Reg.score(x_test,y_test)


# In[104]:


predicted_proba = log_Reg.predict_proba(x_test)
predicted_proba


# In[105]:


#if the probability is >0.5,it places 1and vice versa
predicted_proba[:,1]


# Saving the Model-we use pikle

# 

# In[106]:


import pickle
with open('model','wb') as file:
    pickle.dump(log_Reg,file)


# In[109]:


with open('scaler','wb') as file:
    pickle.dump(scaler,file)


# In[ ]:




