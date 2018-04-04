
# coding: utf-8

# # Hospital Readmissions Data Analysis and Recommendations for Reduction
# 
# ### Background
# In October 2012, the US government's Center for Medicare and Medicaid Services (CMS) began reducing Medicare payments for Inpatient Prospective Payment System hospitals with excess readmissions. Excess readmissions are measured by a ratio, by dividing a hospital’s number of “predicted” 30-day readmissions for heart attack, heart failure, and pneumonia by the number that would be “expected,” based on an average hospital with similar patients. A ratio greater than 1 indicates excess readmissions.
# 
# ### Exercise Directions
# 
# In this exercise, you will:
# + critique a preliminary analysis of readmissions data and recommendations (provided below) for reducing the readmissions rate
# + construct a statistically sound analysis and make recommendations of your own 
# 
# More instructions provided below. Include your work **in this notebook and submit to your Github account**. 
# 
# ### Resources
# + Data source: https://data.medicare.gov/Hospital-Compare/Hospital-Readmission-Reduction/9n3s-kdb3
# + More information: http://www.cms.gov/Medicare/medicare-fee-for-service-payment/acuteinpatientPPS/readmissions-reduction-program.html
# + Markdown syntax: http://nestacms.com/docs/creating-content/markdown-cheat-sheet
# ****

# In[165]:


get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import bokeh.plotting as bkp
from mpl_toolkits.axes_grid1 import make_axes_locatable


# In[166]:


# read in readmissions data provided
hospital_read_df = pd.read_csv('data/cms_hospital_readmissions.csv')


# ****
# ## Preliminary Analysis

# In[167]:


# deal with missing and inconvenient portions of data 
clean_hospital_read_df = hospital_read_df[hospital_read_df['Number of Discharges'] != 'Not Available']
clean_hospital_read_df.loc[:, 'Number of Discharges'] = clean_hospital_read_df['Number of Discharges'].astype(int)
clean_hospital_read_df = clean_hospital_read_df.sort_values('Number of Discharges')


# In[168]:


# generate a scatterplot for number of discharges vs. excess rate of readmissions
# lists work better with matplotlib scatterplot function
x = [a for a in clean_hospital_read_df['Number of Discharges'][81:-3]]
y = list(clean_hospital_read_df['Excess Readmission Ratio'][81:-3])

fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(x, y,alpha=0.2)

ax.fill_between([0,350], 1.15, 2, facecolor='red', alpha = .15, interpolate=True)
ax.fill_between([800,2500], .5, .95, facecolor='green', alpha = .15, interpolate=True)

ax.set_xlim([0, max(x)])
ax.set_xlabel('Number of discharges', fontsize=12)
ax.set_ylabel('Excess rate of readmissions', fontsize=12)
ax.set_title('Scatterplot of number of discharges vs. excess rate of readmissions', fontsize=14)

ax.grid(True)
fig.tight_layout()


# ****
# 
# ## Preliminary Report
# 
# Read the following results/report. While you are reading it, think about if the conclusions are correct, incorrect, misleading or unfounded. Think about what you would change or what additional analyses you would perform.
# 
# **A. Initial observations based on the plot above**
# + Overall, rate of readmissions is trending down with increasing number of discharges
# + With lower number of discharges, there is a greater incidence of excess rate of readmissions (area shaded red)
# + With higher number of discharges, there is a greater incidence of lower rates of readmissions (area shaded green) 
# 
# **B. Statistics**
# + In hospitals/facilities with number of discharges < 100, mean excess readmission rate is 1.023 and 63% have excess readmission rate greater than 1 
# + In hospitals/facilities with number of discharges > 1000, mean excess readmission rate is 0.978 and 44% have excess readmission rate greater than 1 
# 
# **C. Conclusions**
# + There is a significant correlation between hospital capacity (number of discharges) and readmission rates. 
# + Smaller hospitals/facilities may be lacking necessary resources to ensure quality care and prevent complications that lead to readmissions.
# 
# **D. Regulatory policy recommendations**
# + Hospitals/facilties with small capacity (< 300) should be required to demonstrate upgraded resource allocation for quality care to continue operation.
# + Directives and incentives should be provided for consolidation of hospitals and facilities to have a smaller number of them with higher capacity and number of discharges.

# ****
# <div class="span5 alert alert-info">
# ### Exercise
# 
# Include your work on the following **in this notebook and submit to your Github account**. 
# 
# A. Do you agree with the above analysis and recommendations? Why or why not?
#    
# B. Provide support for your arguments and your own recommendations with a statistically sound analysis:
# 
#    1. Setup an appropriate hypothesis test.
#    2. Compute and report the observed significance value (or p-value).
#    3. Report statistical significance for $\alpha$ = .01. 
#    4. Discuss statistical significance and practical significance. Do they differ here? How does this change your recommendation to the client?
#    5. Look at the scatterplot above. 
#       - What are the advantages and disadvantages of using this plot to convey information?
#       - Construct another plot that conveys the same information in a more direct manner.
# 
# 
# 
# You can compose in notebook cells using Markdown: 
# + In the control panel at the top, choose Cell > Cell Type > Markdown
# + Markdown syntax: http://nestacms.com/docs/creating-content/markdown-cheat-sheet
# </div>
# ****

# **A. Do you agree with the above analysis and reccomendations with a statistically sound analysis:**
# 
# The analysis seems to be sound according to the one scatterplot used, but their needs to be more statistical testing done. There is no correlation tests shown but a conclusion about correlation is reached. This analysis also concludes that hospital size is the reason for the number of discharges claiming that a hospital with under a capacity of 300 may not have the resources to provide quality care thus the increase in the number of discharges.
# 
# The data is not cleaned properly which could cause a different conclusion to be reached. Once the data is cleaned it could then be analysed properly and a conclusion could be reached.

# **B. Provide supportfor you arguments and your own recommendations with a statiscally sound analysis**
# 
# 

# In[169]:


clean_hospital_read_df.info()


# In[170]:


# Cleaning the data properly
clean_df= clean_hospital_read_df[np.isfinite(clean_hospital_read_df['Excess Readmission Ratio'])]
clean_df.info()


# **1. Null Hypothesis (H0) and Alternative Hypothesis (H1)**
# 
# * H0= There is no statistical correlation between hospital discharges and readmission rates.
# * H1= There is a statistical correlation between hospital discharges and readmission rates.

# In[171]:


from scipy import stats


# In[172]:


# Correlation and p-Value
x= clean_df['Number of Discharges']
y= clean_df['Excess Readmission Ratio']
pearson= stats.pearsonr(x,y)
print(pearson)


# **3. Correlation and p-Value**
# 
# Using scipy.stats.pearsonr(x,y), we calculated the correlationas -0.097 and the p-Value as 1.22 e-25. This p value would indicate a statistical significance which would support the H1 because it is lower than alpha=0.01.

# In[173]:


import seaborn as sns


# In[174]:


sns.regplot(x,y, fit_reg=True)
plt.title('Scatterplot with Regression of Excess Readmission Ratio vs. Number of Discharges')
plt.show()


#  **4. Discuss statistical vs. practical significance:**
#  
# Statistical significance according to the data indicates that there could be a significant difference in the excess readmission ratio and the number of discharges. The p-value is well under the 0.01 value used to rate our null hypothesis and our alternative hypothesis. Practical significance however shows that because the correlation is very close to zero, we cannot say for certain if the alternative hypothesis is true. 

#  **5. What are the advantages and disadvantages of the Scatterplot above:**
# 
# The original scatter plot is useful to determine what the relationship is between the data. A better plot would have inclueded the regession line to indicate the type of correlation that existed between the data.
