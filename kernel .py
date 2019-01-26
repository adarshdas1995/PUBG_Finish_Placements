
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# ****** This code was run on kaggle notebook for PUBG Finish Placement Prediction  ******
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
df = pd.read_csv('../input/train_V2.csv')
# Any results you write to the current directory are saved as output.


# In[2]:


# Dropping ID columns. Not useful for our analysis
df = df.drop(columns = ['Id', 'groupId','matchId'])


# In[3]:


# Used to display all columns
pd.set_option('display.max_columns', 500) 
df.head()


# In[4]:


# Certain modes have very low number of players
# We are only considering the 6 most popular modes. Dropping 'normal-squad-fpp','crashfpp','normal-duo-fpp','flaretpp','normal-solo-fpp','flarefpp','normal-squad','normal-solo','normal-duo','crashtpp'.
df = df.set_index('matchType')
df = df.drop(['normal-squad-fpp','crashfpp','normal-duo-fpp','flaretpp','normal-solo-fpp','flarefpp','normal-squad','normal-solo','normal-duo','crashtpp' ])
df = df.reset_index()


# In[5]:


# Checking correlation amongst all the features. Highly un-correlated features will be removed
import matplotlib.pyplot as plt
corr = df.corr()
corr.style.background_gradient()


# In[6]:


# We will now drop uncorrelated features 'killPoints','matchDuration','maxPlace','numGroups','rankPoints','roadKills','swimDistance','vehicleDestroys','killPoints','winPoints'. 
df = df.drop(columns = ['killPoints','matchDuration','maxPlace','numGroups','rankPoints','roadKills','swimDistance','vehicleDestroys','killPoints','winPoints'])


# In[7]:


df.head()
# The matchType feature is the only one with 'object' datatype. We will use LabelEncoder() to label encode it.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['matchType'] = le.fit_transform(df['matchType'])


# In[8]:


# We are trying to figure out if any of the data has skewness to it. Any skewed data has to be normalised.
# We are also trying to bring down are data between [0,1] so that the model is better able to understand it.
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
df.hist(bins = 50, figsize = (20,15))
plt.show()


# In[9]:


import numpy as np
df['damageDealt'] = df['damageDealt'].apply(np.cbrt)
df['walkDistance'] = df['walkDistance'].apply(np.cbrt)
df['rideDistance'] = df['rideDistance'].apply(np.cbrt)
df['longestKill'] = df['longestKill'].apply(np.cbrt)


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
df.hist(bins = 50, figsize = (20,15))
plt.show()


# In[11]:


df['walkDistance'] = df['walkDistance']/np.max(df['walkDistance'])
df['rideDistance'] = df['rideDistance']/np.max(df['rideDistance'])
df['damageDealt'] = df['damageDealt']/np.max(df['damageDealt'])
df['weaponsAcquired'] = df['weaponsAcquired'].apply(np.cbrt)
df['kills'] = df['kills'].apply(np.cbrt)


# In[12]:


df.hist(bins = 50, figsize = (20,15))
plt.show()


# In[13]:


# The killPlace attribute shows what is the players position on the kills leaderboard.
# Since it's a form of ranking from 1-100
# We group them into 10 bands each of width 10
df['killPlaceBand'] = pd.cut(df['killPlace'],10)
df[['killPlaceBand', 'winPlacePerc']].groupby(['killPlaceBand'], as_index = False).mean().sort_values(by = 'killPlaceBand', ascending = True)


# In[14]:


df['killPlaceBand'] = le.fit_transform(df['killPlaceBand'])


# In[15]:


df = df.drop(columns = 'killPlace')


# In[16]:


# The dataset is too big. Any rows which contain missing values are less. So we just dropped them.
df = df.dropna()


# In[17]:


# Rather than using a separate test.csv, we split our train data into test and train.
# It's fairly big enough to not cause any issues.
from sklearn.model_selection import train_test_split
train, test = train_test_split(df,train_size = 0.66)


# In[18]:


x_train = train.drop(columns = 'winPlacePerc')
y_train = train['winPlacePerc']
x_test = test.drop(columns = 'winPlacePerc')
y_test = test['winPlacePerc']


# In[19]:


# Linear Regressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
l_r = LinearRegression()
l_r.fit(x_train,y_train)
y_pred_l_r = l_r.predict(x_test)
print(l_r.score(x_train,y_train))
print(l_r.score(x_test,y_test))
print(mean_absolute_error(y_test,y_pred_l_r))


# In[20]:


# XGBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
xgb = XGBRegressor()
xgb.fit(x_train,y_train)
xgb.score(x_train,y_train)
xgb.score(x_test,y_test)
y_pred_xgb = xgb.predict(x_test)
print(mean_absolute_error(y_test,y_pred_xgb))


# In[21]:


# Cross Validation
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
xgb = XGBRegressor()
scores = cross_val_score(xgb,x_train,y_train,scoring = 'neg_mean_absolute_error',cv = 10)
print(scores)
print(scores.mean())


# In[22]:


# Decision tree
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(max_depth = 6)
dtr.fit(x_train,y_train)
dtr.score(x_train,y_train)
dtr.score(x_test,y_test)
y_pred_dtr = dtr.predict(x_test)
print(mean_absolute_error(y_test,y_pred_dtr))
print(dtr.score(x_train,y_train))
print(dtr.score(x_test,y_test))


# In[23]:


# SGDRegressor
from sklearn import linear_model
sgd = linear_model.SGDRegressor(max_iter=1000)
sgd.fit(x_train,y_train)
print(sgd.score(x_train,y_train))
print(sgd.score(x_test,y_test))
y_pred_sgd = sgd.predict(x_test)
print(mean_absolute_error(y_test,y_pred_sgd))


# In[24]:


#Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
r_f_r = RandomForestRegressor(n_estimators = 10)
r_f_r.fit(x_train,y_train)
r_f_r.score(x_train,y_train)
y_pred = r_f_r.predict(x_test)
print(mean_absolute_error(y_test,y_pred))
print(r_f_r.score(x_test,y_test))


# In[25]:


mae_lr = mean_absolute_error(y_test,y_pred_l_r)
mae_xgb = mean_absolute_error(y_test,y_pred_xgb)
mae_sgd = mean_absolute_error(y_test,y_pred_sgd)
mae_dtr = mean_absolute_error(y_test,y_pred_dtr)
mae_rfr = mean_absolute_error(y_test,y_pred)
                              


# In[26]:


models = pd.DataFrame({
    'Model': ['Linear Regression', 'XGBoost Regressor', 'Decision Tree', 
              'Random Forest','Stochastic Gradient Decent'],
    'MAE': [mae_lr,mae_xgb,mae_dtr,mae_rfr,mae_sgd]})
models.sort_values(by='MAE', ascending=True)


# In[28]:


# XGBoost Regressor gives us the best MAE
# Pickling
import pickle
filename = 'xgb_pickle.sav'
pickle.dump(xgb,open(filename,'wb'))

