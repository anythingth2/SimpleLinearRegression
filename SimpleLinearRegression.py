
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


# In[98]:


mean = y.mean()
print('Mean =',mean.value)

median = y.median()
print('Median =',median.value)

mode = y.mode()
print('Mode at',mode.values[0][0])

standardDeviation = y.std()
print('Standard Deviation =',standardDeviation.value)


# In[118]:


y.plot()


# In[114]:


y.plot.hist()


# In[115]:


y.plot.box()


# In[117]:





# In[123]:



ax = fig.add_subplot(111)
ax.set_xlabel('time')
ax.set_ylabel('number of users')
plt.scatter(x,y,color='lightblue')
plt.plot(x,y_pred,color='red')


# In[71]:


plt.show()

