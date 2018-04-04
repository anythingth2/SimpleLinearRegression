
# coding: utf-8

# In[65]:


import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[66]:


dataframe = pd.read_csv('WWWusage.csv')


# In[61]:


x = pd.DataFrame(dataframe,columns=['time'])
y = pd.DataFrame(dataframe,columns=['value'])


# In[62]:


rgs = linear_model.LinearRegression()
rgs.fit(x,y)


# In[63]:


y_pred = rgs.predict(x)


# In[70]:


fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('time')
ax.set_ylabel('number of users')
plt.scatter(x,y,color='red')
plt.scatter(x,y_pred,color='blue')


# In[71]:


plt.show()

