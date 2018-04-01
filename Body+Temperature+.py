
# coding: utf-8

# In[187]:


import pandas as pd


# In[188]:


temp='human_body_temperature.csv'
df=pd.read_csv(temp)
print(df)


# In[189]:


df2=df['temperature']


# In[190]:


import numpy as np
import matplotlib.pyplot as plt


# In[191]:


mean=np.mean(df2)
std=np.std(df2)


# In[192]:


def ecdf(data):
    """compute ecdf for a one dimensional array of measurments"""
    #number of data point:n
    n=len(data)
    
    #x-data for ecdf:x
    x=np.sort(data)
       
    #y-data for ecdf:y
    y= np.arange(1, n+1)/n
    
    return x,y 


# In[193]:


samples= np.random.normal(mean, std, size=10000)
x_theo, y_theo=ecdf(samples)
x, y =ecdf(df2)
plt.plot(x_theo, y_theo, marker='.', linestyle='none')
plt.plot(x, y, marker='.', linestyle='none')
plt.xlabel('temperature')
plt.ylabel('ECDF')
plt.margins(0.02)
plt.show()


# In[194]:


df2_mean= np.mean(df2)
print(df2_mean)


# In[195]:


conf= np.percentile(df2_mean, [2.5,97.5])
print(conf)


# In[196]:


for _ in range(1000):
    #generate a bootstrap sample
    
    bs_sample=np.random.choice(df2, size=len(df2))
    #compute and plot ecdf
    
    x, y=ecdf(bs_sample)
    _=plt.plot(x,y, marker='.', linestyle='none', color='gray', alpha=0.10)
    
#compute ecdf from original data
x, y=ecdf(df2)
_=plt.plot(x, y, marker='.')

#makemargins and labels
plt.margins(0.02)
_=plt.xlabel('Temperature (F)')
_=plt.ylabel('ECDF')
plt.show()
    
             


# In[197]:


def bootstrapes_replicate_1d(data, func):
    return func(np.random.choice(data, size=len(data)))


# In[198]:


def draw_bs_reps(data, func, size=1):
    
    bs_replicates= np.empty(data, size=len(data))
    
    for i in range(size):
        bs_replicates[i]= bootstraps_replicate_1d(data, func)
    return bs_replicates


# In[199]:


from scipy import stats


# In[200]:


stats.ttest_1samp(df2, 98.6)

