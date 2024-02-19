#!/usr/bin/env python
# coding: utf-8

# # Duzce_database_damage_classifier

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# # Load_data

# In[2]:


headers = ['Construction_time', 'Storey_height', 'Number_of_stories', 'Floor_area', 'Plan_irregularity_A1', 'Plan_irregularity_A2', 'Plan_irregularity_A3', 'Plan_irregularity_A4', 'Vertical_irregularity_B1', 'Vertical_irregularity_B2', 'Vertical_irregularity_B3', 'Bays_Number_x', 'Bays_Number_y', 'System_type', 'Overhanging_area', 'Column_area', 'storey_number', 'Class']
df = pd.read_csv('Database.csv', names=headers)


# # Explore Data

# In[3]:


df.dtypes


# In[4]:


df.describe(include='all')


# In[5]:


df.info


# In[6]:


df['Class'].nunique()


# In[7]:


df['Class'].unique()


# # Data Preprocessing or Cleaning
# 

# # Dealing with missing data

# In[8]:


# Replacing non-standard value with 'Nan' so that pandas dataframe can ditect

df.replace('NA', np.nan, inplace = True)
df.replace('N/L', np.nan, inplace = True)
df.replace(0, np.nan, inplace = True)


# In[9]:


df.describe(include='all')


# In[10]:


# Dealing with those variables which need to be replaced by mean

from sklearn.impute import SimpleImputer
imputer_mean = SimpleImputer(missing_values = np.nan, strategy = 'mean')
df[['Construction_time']] = imputer_mean.fit_transform(df[['Construction_time']]).astype('object')

#df[['feature', 'Feature', 'Feature']] = imputer_mean.fit_transform(df[['feature', 'Feature', 'Feature']]).astype('object')
#df.head(10)


# Dealing with those variables which need to be replaced by median

imputer_mean = SimpleImputer(missing_values = np.nan, strategy = 'median')
df[['Overhanging_area', 'Column_area']] = imputer_mean.fit_transform(df[['Overhanging_area', 'Column_area']]).astype('object')


# In[11]:


# Replacing null values with most frequent values in case for categorical variable (mode)

imputer_most_freq = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
df[['Bays_Number_x', 'Bays_Number_y', 'Plan_irregularity_A1', 'Plan_irregularity_A2', 'Plan_irregularity_A3', 'Plan_irregularity_A4', 'Vertical_irregularity_B1', 'Vertical_irregularity_B2', 'Vertical_irregularity_B3']] = imputer_most_freq.fit_transform(df[['Bays_Number_x', 'Bays_Number_y', 'Plan_irregularity_A1', 'Plan_irregularity_A2', 'Plan_irregularity_A3', 'Plan_irregularity_A4', 'Vertical_irregularity_B1', 'Vertical_irregularity_B2', 'Vertical_irregularity_B3']])


# In[12]:


# Drop whole row with NaN in 'Class' column 
df.dropna(subset = ['Class'], axis=0, inplace=True)

# Reset index because of dropping rows
df.reset_index(drop = True, inplace = True)


# In[13]:


#df('Class').value_counts()
df['Class'].unique()
df.describe(include='all')




# # Dealing with outliers

# In[14]:


plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df['Floor_area'])

plt.subplot(1,2,2)
sns.distplot(df['Overhanging_area'])

plt.show()


# In[15]:


plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df['Column_area'])

plt.subplot(1,2,2)
sns.distplot(df['Storey_height'])

plt.show()


# In[16]:


plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df['Construction_time'])

plt.subplot(1,2,2)
sns.distplot(df['Number_of_stories'])

plt.show()


# In[17]:


plt.figure(figsize=(16,5))
plt.subplot(1,2,1)
sns.distplot(df['storey_number'])


plt.show()


# In[ ]:





# # Outlier Detection
# # Z score method

# In[18]:


# For normally distributed feature

print("Mean value of feature", df['Storey_height'].mean())
print("Std value of feature", df['Storey_height'].std())
print("Min value of feature", df['Storey_height'].min())
print("Max value of feature", df['Storey_height'].max())


# In[19]:


# Find the boundaries values
# Based on Z score calculation

print("Highest allowed", df['Storey_height'].mean() + 3*df['Storey_height'].std())
print("Lowest allowed", df['Storey_height'].mean() - 3*df['Storey_height'].std())


# In[20]:


# Find the outliers

df[(df['Storey_height']>3.2481512971316526) | (df['Storey_height'] < 2.426288536893243)]


#  # IQR method (Interquartile Range)

# In[21]:


# For skewed feature (Floor_area)

df['Floor_area'].describe()


# In[22]:


sns.boxplot(df['Floor_area'])


# In[23]:


# Finding the IQR

percentile25 = df['Floor_area'].quantile(0.25)
percentile75 = df['Floor_area'].quantile(0.75)


# In[24]:


percentile25


# In[25]:


percentile75


# In[26]:


iqr = percentile75 - percentile25 


# In[27]:


iqr


# In[28]:


# Indicating the limit values

upper_limit2 = percentile75 + 1.5*iqr
lower_limit2 = percentile25 - 1.5*iqr


# In[29]:


print('Upper limit', upper_limit2)
print('Lower limit', lower_limit2)


# In[30]:


# Find the outliers

df[df['Floor_area'] > upper_limit2]


# In[31]:


df[df['Floor_area'] < lower_limit2]


# In[32]:


# For skewed feature (Column_area)

#df['Column_area'].describe()


# In[33]:


#sns.boxplot(df['Column_area'])


# In[34]:


# Finding the IQR

#percentile25_2 = df['Column_area'].quantile(0.25)
#percentile75_2 = df['Column_area'].quantile(0.75)


# In[35]:


#iqr = percentile75_2 - percentile25_2 


# In[36]:


# Indicating the limit values

#upper_limit3 = percentile75_2 + 1.5*iqr
#lower_limit3 = percentile25_2 - 1.5*iqr


# In[37]:


#print('Upper limit', upper_limit3)
#print('Lower limit', lower_limit3)


# In[38]:


# Find the outliers

#df[df['Column_area'] > upper_limit3]


# In[39]:


#df[df['Column_area'] < lower_limit3]


# In[40]:


# For skewed feature (Overhanging_area)

#df['Overhanging_area'].describe()


# In[41]:


#sns.boxplot(df['Overhanging_area'])


# In[42]:


# Finding the IQR

#percentile25_3 = df['Overhanging_area'].quantile(0.25)
#percentile75_3 = df['Overhanging_area'].quantile(0.75)


# In[43]:


#iqr = percentile75_3 - percentile25_3 


# In[44]:


# Indicating the limit values

#upper_limit4 = percentile75_3 + 1.5*iqr
#lower_limit4 = percentile25_3 - 1.5*iqr


# In[45]:


#print('Upper limit', upper_limit4)
#print('Lower limit', lower_limit4)


# In[46]:


# Find the outliers

#df[df['Overhanging_area'] > upper_limit4]


# In[47]:


#df[df['Overhanging_area'] < lower_limit4]


# # Outlier Removal for Normally Distributed Feature
# # Trimming

# In[48]:


#df = df[(df['Feature'] < value) & (df['Feature'] > value)] # values are highest and lowest allowed
#df


# #  or Capping

# In[49]:


# Alternate method

upper_limit = df['Storey_height'].mean() + 3*df['Storey_height'].std()
lower_limit = df['Storey_height'].mean() - 3*df['Storey_height'].std()


# In[50]:


df['Storey_height'] = np.where(
    df['Storey_height']> upper_limit, 
    upper_limit, 
    np.where(
        df['Storey_height'] < lower_limit,
        lower_limit, 
        df['Storey_height']
    )
)



# In[51]:


df['Storey_height'].describe()


# # Outlier Removal for Skewed  Feature

# # Trimming

# In[52]:


#df = df[df['Feauture'] < upper_limit2]


# In[53]:


#df.shape()


# # or Capping

# In[54]:


# Floor_area

df['Floor_area'] = np.where(
    df['Floor_area']> upper_limit2, 
    upper_limit2, 
    np.where(
        df['Floor_area'] < lower_limit2,
        lower_limit2, 
        df['Floor_area']
    )
)


# In[55]:


df['Floor_area'].describe()


# In[56]:


# Column_area

#df['column_area'] = np.where(
    #df['Column_area']> upper_limit3, 
    #upper_limit3, 
    #np.where(
        #df['Column_area'] < lower_limit3,
        #lower_limit3, 
        #df['Column_area']
    #)
#)


# In[57]:


#df['Column_area'].describe()


# In[58]:


# Overhanging_area

#df['Overhanging_area'] = np.where(
    #df['Overhanging_area']> upper_limit4, 
    #upper_limit4, 
    #np.where(
        #df['Overhanging_area'] < lower_limit4,
        #lower_limit4, 
        #df['Overhanging_area']
    #)
#)


# In[59]:


#df['Overhanging_area'].describe()


# # Creating the Random Forest Model

# # Preparing Data

# In[60]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier


# In[61]:


# Select inputs and target
#X = df[['Input', 'Input', 'Input']]
#y = df['Class']


# Location by index
X = df.iloc[:,:-1]
y = df.iloc[:,-1]


# In[62]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 56)


# # Dealing with Categorical Values

# # Label Encoding

# In[63]:


# Used for musticlass target variable.

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[64]:


le.fit(y_train)


# In[65]:


le.classes_


# In[66]:


y_train = le.transform(y_train)
y_test = le.transform(y_test)


# In[67]:


y_train


# # Ordinal Encoding

# In[68]:


# Categorical values have relation between them. Means come in order.
# Ordinal categorical data.

#from sklearn.preprocessing import OrdinalEncoder
#oe =  OrdinalEncoder(categories = [['Poor','Average','Good'],['Good','Better','Best'])


# In[69]:


#oe.fit(X_train)


# In[70]:


#X_train = oe.transform(X_train)
#X_test = oe.transform(X_test)


# In[71]:


#X_train


# In[72]:


#oe.categories_


# # One-hot encoding

# In[73]:


# Multiclass catergorical value. For example city of a country.
# Nominal categorical data

#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoder
#ct = ColumnTransformer(transformers =[('encoder',  OneHotEncoder(), [0, 2, 3])], remainder = 'passthrough')
#df = pd.DataFrame(ct.fit_transform(df))
#df.columns = ['Feature', 'Feature', 'Feature']


# # Feature Scaling (Standardization)

# In[74]:


# Feature scaling shoud be done after data splitting.There should be no influence on test data.

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train.iloc[:,:-1] = sc.fit_transform(X_train.iloc[:,:-1])

# Not using fit method so that model doesn't know the mean and standard deviation of the test dataset
X_test.iloc[:,:-1] = sc.transform(X_test.iloc[:,:-1])


# # Creating Model

# In[75]:


clf = RandomForestClassifier(max_samples=0.75,random_state=42)


# In[76]:


clf.fit(X_train, y_train)


# In[77]:


y_pred = clf.predict(X_test)


# In[78]:


accuracy_score(y_test, y_pred)


# # Cross Validation

# In[79]:


# For class imbalanced dataset applying stratified K-Fold cross validation

from sklearn.model_selection import cross_val_score
np.mean(cross_val_score(RandomForestClassifier(),X,y,cv=10,scoring='accuracy'))


# # Hyperparameter Tuning

# # GridSearchCV

# In[80]:


# Number of trees in random forest
n_estimators = [20,60,100,120]

# Number of features to consider at every split
max_features = [0.2,0.6,1.0]

# Maximum number of levels in tree
max_depth = [2,8,None]

# Number of samples
max_samples = [0.5,0.75,1.0]



# In[81]:


param_grid = {'n_estimators': n_estimators,
             'max_features': max_features,
             'max_depth': max_depth,
             'max_samples': max_samples
             }
print(param_grid)



# In[82]:


rf = RandomForestClassifier()


# In[83]:


from sklearn.model_selection import GridSearchCV

rf_grid = GridSearchCV(estimator = rf,
                      param_grid = param_grid,
                      cv = 5,
                      verbose = 2,
                      n_jobs = -1)


# In[84]:


rf_grid.fit(X_train, y_train)


# In[85]:


rf_grid.best_params_


# In[86]:


rf_grid.best_score_


# # Model Evaluation

# In[87]:


# Classification Report

print(classification_report(y_test, y_pred))


# In[88]:


# Confusion Matrix

cf_matrix = confusion_matrix(y_test, y_pred)
print(cf_matrix)


# In[89]:


labels = ['C/R', 'L', 'M', 'N', 'S']
labels.sort()


# In[90]:


import seaborn as sns

fig = plt.figure(figsize=(10,10))
ax = sns.heatmap(cf_matrix, annot=True, cmap='coolwarm', fmt='.0f',
                xticklabels=labels, yticklabels=labels)

ax.set_title('Seaborn Confuse Matrix with labels\n\n');
ax.set_xlabel('\nPredicted values')
ax.set_ylabel('Actual Values');


# # Feature Importance

# In[92]:


# Correlation Matrix

#corr = df.corr()
#corr


# In[ ]:


# Display Pearson Correlation Matrix in Heatmap

corr = df.corr()
plt.figure(figsize=(14,10))
sns.heatmap(corr, annot = True, cmap = 'coolwarm')


# In[ ]:





# In[ ]:





# In[ ]:




