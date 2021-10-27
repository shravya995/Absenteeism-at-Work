
# coding: utf-8

# In[1]:


#Load libraries
import os
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
from random import randrange, uniform
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn.cross_validation import train_test_split


# In[2]:


#Set working directory
os.chdir("C:/Users/SHRAVYA/Desktop/edwisor/project 1")


# In[3]:


#Load data
Absenteeism_at_work = pd.read_csv("Absenteeism_at_work_Project.csv")


# In[ ]:


#----------------------------------PRE PROCESSING-EXPLORATORY DATA ANALYSIS----------------------------------------------------#


# In[4]:


#Exploratory Data Analysis
Absenteeism_at_work['Reason for absence']=Absenteeism_at_work['Reason for absence'].astype(object)
Absenteeism_at_work['Month of absence']=Absenteeism_at_work['Month of absence'].astype(object)
Absenteeism_at_work['Day of the week']=Absenteeism_at_work['Day of the week'].astype(object)
Absenteeism_at_work['Seasons']=Absenteeism_at_work['Seasons'].astype(object)
Absenteeism_at_work['Service time']=Absenteeism_at_work['Service time'].astype(object)
Absenteeism_at_work['Hit target']=Absenteeism_at_work['Hit target'].astype(object)
Absenteeism_at_work['Disciplinary failure']=Absenteeism_at_work['Disciplinary failure'].astype(object)
Absenteeism_at_work['Education']=Absenteeism_at_work['Education'].astype(object)
Absenteeism_at_work['Son']=Absenteeism_at_work['Son'].astype(object)
Absenteeism_at_work['Social drinker']=Absenteeism_at_work['Social drinker'].astype(object)
Absenteeism_at_work['Social smoker']=Absenteeism_at_work['Social smoker'].astype(object)
Absenteeism_at_work['Pet']=Absenteeism_at_work['Pet'].astype(object)


# In[ ]:


#---------------------------------------MISSING VALUE ANALYSIS-----------------------------------------------------------------#


# In[5]:


#Create dataframe with missing percentage
missing_val = pd.DataFrame(Absenteeism_at_work.isnull().sum())

#Reset index
missing_val = missing_val.reset_index()

#Rename variable
missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})

#Calculate percentage
missing_val['Missing_percentage'] = (missing_val['Missing_percentage']/len(Absenteeism_at_work))*100

#descending order
missing_val = missing_val.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)

#save output results 
missing_val.to_csv("Missing_perc.csv", index = False)


# In[6]:


#KNN imputation
#Assigning levels to the categories
lis = []
for i in range(0, Absenteeism_at_work.shape[1]):
    #print(i)
    if(Absenteeism_at_work.iloc[:,i].dtypes == 'object'):
        Absenteeism_at_work.iloc[:,i] = pd.Categorical(Absenteeism_at_work.iloc[:,i])
        #print(marketing_train[[i]])
        Absenteeism_at_work.iloc[:,i] = Absenteeism_at_work.iloc[:,i].cat.codes 
        Absenteeism_at_work.iloc[:,i] = Absenteeism_at_work.iloc[:,i].astype('object')
        
        lis.append(Absenteeism_at_work.columns[i])


# In[7]:


#replace -1 with NA to impute
for i in range(0, Absenteeism_at_work.shape[1]):
    Absenteeism_at_work.iloc[:,i] = Absenteeism_at_work.iloc[:,i].replace(-1, np.nan) 


# In[8]:


#Impute with median
Absenteeism_at_work['Absenteeism time in hours'] = Absenteeism_at_work['Absenteeism time in hours'].fillna(Absenteeism_at_work['Absenteeism time in hours'].median())
Absenteeism_at_work['Body mass index'] = Absenteeism_at_work['Body mass index'].fillna(Absenteeism_at_work['Body mass index'].median())
Absenteeism_at_work['Height'] = Absenteeism_at_work['Height'].fillna(Absenteeism_at_work['Height'].median())
Absenteeism_at_work['Weight'] = Absenteeism_at_work['Weight'].fillna(Absenteeism_at_work['Weight'].median())
Absenteeism_at_work['Pet'] = Absenteeism_at_work['Pet'].fillna(Absenteeism_at_work['Pet'].median())
Absenteeism_at_work['Social smoker'] = Absenteeism_at_work['Social smoker'].fillna(Absenteeism_at_work['Social smoker'].median())
Absenteeism_at_work['Social drinker'] = Absenteeism_at_work['Social drinker'].fillna(Absenteeism_at_work['Social drinker'].median())
Absenteeism_at_work['Son'] = Absenteeism_at_work['Son'].fillna(Absenteeism_at_work['Son'].median())
Absenteeism_at_work['Education'] = Absenteeism_at_work['Education'].fillna(Absenteeism_at_work['Education'].median())
Absenteeism_at_work['Disciplinary failure'] = Absenteeism_at_work['Disciplinary failure'].fillna(Absenteeism_at_work['Disciplinary failure'].median())
Absenteeism_at_work['Hit target'] = Absenteeism_at_work['Hit target'].fillna(Absenteeism_at_work['Hit target'].median())
Absenteeism_at_work['Age'] = Absenteeism_at_work['Age'].fillna(Absenteeism_at_work['Age'].median())
Absenteeism_at_work['Service time'] = Absenteeism_at_work['Service time'].fillna(Absenteeism_at_work['Service time'].median())
Absenteeism_at_work['Distance from Residence to Work'] = Absenteeism_at_work['Distance from Residence to Work'].fillna(Absenteeism_at_work['Distance from Residence to Work'].median())
Absenteeism_at_work['Transportation expense'] = Absenteeism_at_work['Transportation expense'].fillna(Absenteeism_at_work['Transportation expense'].median())
Absenteeism_at_work['Month of absence'] = Absenteeism_at_work['Month of absence'].fillna(Absenteeism_at_work['Month of absence'].median())
Absenteeism_at_work['Reason for absence'] = Absenteeism_at_work['Reason for absence'].fillna(Absenteeism_at_work['Reason for absence'].median())
Absenteeism_at_work['Work load Average/day '] = Absenteeism_at_work['Work load Average/day '].fillna(Absenteeism_at_work['Work load Average/day '].median())


# In[9]:


Absenteeism_at_work.isnull().sum()


# In[12]:


Absenteeism_at_work = Absenteeism_at_work.dropna(how='all')


# In[13]:


Absenteeism_at_work.isnull().sum()


# In[10]:


cnames =  ["ID", "Transportation expense", "Distance from Residence to Work", "Age", "Height", "Body mass index", "Absenteeism time in hours"]


# In[ ]:


#----------------------------------------FEATURE SELECTION--------------------------------------------------------------------#


# In[11]:


##Correlation analysis
#Correlation plot
df_corr = Absenteeism_at_work.loc[:,cnames]


# In[12]:


#Set the width and hieght of the plot
f, ax = plt.subplots(figsize=(7, 5))

#Generate correlation matrix
corr = df_corr.corr()

#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
plt.savefig('correlation.png')


# In[13]:


#Chisquare test of independence
#Save categorical variables
cat_names = ["Reason for absence", "Month of absence", "Day of the week", "Seasons", "Service time", "Hit target", "Disciplinary failure", "Education", "Son", "Social drinker","Social smoker","Pet"]


# In[14]:


#loop for chi square values
for i in cat_names:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(Absenteeism_at_work['Absenteeism time in hours'], Absenteeism_at_work[i]))
    print(p)


# In[ ]:


#----------------------------------------------FEATURE SCALING----------------------------------------------------------------#


# In[19]:


#feature reduction
Absenteeism_at_work = Absenteeism_at_work.drop(['Weight', 'Hit target', 'Education', 'Social smoker', 'Pet'], axis=1)


# In[15]:


#Nomalisation
for i in cnames:
    print(i)
    Absenteeism_at_work[i] = (Absenteeism_at_work[i] - min(Absenteeism_at_work[i]))/(max(Absenteeism_at_work[i]) - min(Absenteeism_at_work[i]))


# In[ ]:


#------------------------------------------DATA SAMPLING-----------------------------------------------------------------------#


# In[16]:


#Divide data into train and test
train, test = train_test_split(Absenteeism_at_work, test_size=0.25, random_state=42)


# In[ ]:


#---------------------------------------------MODELLING-------------------------------------------------------------------------#


# In[17]:


# Decision Tree


# In[18]:


#Decision tree for regression
fit_DT = DecisionTreeRegressor(max_depth=2).fit(train.iloc[:,0:9], train.iloc[:,9])


# In[19]:


#checking for any missing valuses that has leeked in
np.where(Absenteeism_at_work.values >= np.finfo(np.float64).max)


# In[20]:


np.isnan(Absenteeism_at_work.values.any())


# In[29]:


test = test.fillna(train.mean())


# In[23]:


#Decision tree for regression
fit_DT = DecisionTreeRegressor(max_depth=2).fit(train.iloc[:,0:15], train.iloc[:,15])


# In[24]:


Absenteeism_at_work.shape


# In[25]:


#Apply model on test data
predictions_DT = fit_DT.predict(test.iloc[:,0:15])


# In[28]:


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


# In[29]:


rmse(test.iloc[:,15], predictions_DT)


# In[30]:


#rmse value using decision tree is 0.225944999314018


# In[37]:


#Divide data into train and test
X = Absenteeism_at_work.values[:, 0:15]
Y = Absenteeism_at_work.values[:,15]

X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2)


# In[38]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier

RF_model = RandomForestClassifier(n_estimators = 20).fit(X_train, y_train)


# In[39]:


RF_Predictions = RF_model.predict(X_test)


# In[ ]:


#------------------------------------------PLOTS OF VARIABLES------------------------------------------------------------------#


# In[43]:


#plots
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[44]:


import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


# In[45]:


np.random.seed(sum(map(ord, "categorical")))


# In[46]:


Absenteeism_at_work.columns


# In[71]:


sns.stripplot(x="Body mass index", y="Absenteeism time in hours", data=Absenteeism_at_work);
plt.savefig('Body mass index.png')


# In[36]:


sns.stripplot(x="Reason for absence", y="Absenteeism time in hours", data=Absenteeism_at_work);
plt.savefig('Reason for absence.png')


# In[41]:


sns.stripplot(x="Month of absence", y="Absenteeism time in hours", data=Absenteeism_at_work);
plt.savefig('Month of absence.png')


# In[42]:


sns.stripplot(x="Day of the week", y="Absenteeism time in hours", data=Absenteeism_at_work);
plt.savefig('Day of the week.png')


# In[43]:


sns.stripplot(x="Seasons", y="Absenteeism time in hours", data=Absenteeism_at_work);
plt.savefig('Seasons.png')


# In[44]:


sns.stripplot(x="Transportation expense", y="Absenteeism time in hours", data=Absenteeism_at_work);
plt.savefig('Transportation expense.png')


# In[45]:


sns.stripplot(x="Distance from Residence to Work", y="Absenteeism time in hours", data=Absenteeism_at_work);
plt.savefig('Distance from Residence to Work.png')


# In[46]:


sns.stripplot(x="Service time", y="Absenteeism time in hours", data=Absenteeism_at_work);
plt.savefig('Service time.png')


# In[47]:


sns.stripplot(x="Age", y="Absenteeism time in hours", data=Absenteeism_at_work);
plt.savefig('Age.png')


# In[48]:


sns.stripplot(x="Disciplinary failure", y="Absenteeism time in hours", data=Absenteeism_at_work);
plt.savefig('Disciplinary failure.png')


# In[49]:


sns.stripplot(x="Son", y="Absenteeism time in hours", data=Absenteeism_at_work);
plt.savefig('Son.png')


# In[50]:


sns.stripplot(x="Social drinker", y="Absenteeism time in hours", data=Absenteeism_at_work);
plt.savefig('Social drinker.png')


# In[51]:


sns.stripplot(x="Height", y="Absenteeism time in hours", data=Absenteeism_at_work);
plt.savefig('Height.png')

