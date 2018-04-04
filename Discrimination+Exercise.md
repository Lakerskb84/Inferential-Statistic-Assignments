
# Examining Racial Discrimination in the US Job Market

### Background
Racial discrimination continues to be pervasive in cultures throughout the world. Researchers examined the level of racial discrimination in the United States labor market by randomly assigning identical résumés to black-sounding or white-sounding names and observing the impact on requests for interviews from employers.

### Data
In the dataset provided, each row represents a resume. The 'race' column has two values, 'b' and 'w', indicating black-sounding and white-sounding. The column 'call' has two values, 1 and 0, indicating whether the resume received a call from employers or not.

Note that the 'b' and 'w' values in race are assigned randomly to the resumes when presented to the employer.

<div class="span5 alert alert-info">
### Exercises
You will perform a statistical analysis to establish whether race has a significant impact on the rate of callbacks for resumes.

Answer the following questions **in this notebook below and submit to your Github account**. 

   1. What test is appropriate for this problem? Does CLT apply?
   2. What are the null and alternate hypotheses?
   3. Compute margin of error, confidence interval, and p-value. Try using both the bootstrapping and the frequentist statistical approaches.
   4. Write a story describing the statistical significance in the context or the original problem.
   5. Does your analysis mean that race/name is the most important factor in callback success? Why or why not? If not, how would you amend your analysis?

You can include written notes in notebook cells using Markdown: 
   - In the control panel at the top, choose Cell > Cell Type > Markdown
   - Markdown syntax: http://nestacms.com/docs/creating-content/markdown-cheat-sheet


#### Resources
+ Experiment information and data source: http://www.povertyactionlab.org/evaluation/discrimination-job-market-united-states
+ Scipy statistical methods: http://docs.scipy.org/doc/scipy/reference/stats.html 
+ Markdown syntax: http://nestacms.com/docs/creating-content/markdown-cheat-sheet
+ Formulas for the Bernoulli distribution: https://en.wikipedia.org/wiki/Bernoulli_distribution
</div>
****


```python
import pandas as pd
import numpy as np
from scipy import stats
```


```python
data = pd.io.stata.read_stata('data/us_job_market_discrimination.dta')
```


```python
# number of callbacks for black-sounding names
sum(data[data.race=='w'].call)
```




    235.0




```python
data.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>ad</th>
      <th>education</th>
      <th>ofjobs</th>
      <th>yearsexp</th>
      <th>honors</th>
      <th>volunteer</th>
      <th>military</th>
      <th>empholes</th>
      <th>occupspecific</th>
      <th>...</th>
      <th>compreq</th>
      <th>orgreq</th>
      <th>manuf</th>
      <th>transcom</th>
      <th>bankreal</th>
      <th>trade</th>
      <th>busservice</th>
      <th>othservice</th>
      <th>missind</th>
      <th>ownership</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>b</td>
      <td>1</td>
      <td>4</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>17</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td></td>
    </tr>
    <tr>
      <th>1</th>
      <td>b</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>316</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td></td>
    </tr>
    <tr>
      <th>2</th>
      <td>b</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td></td>
    </tr>
    <tr>
      <th>3</th>
      <td>b</td>
      <td>1</td>
      <td>3</td>
      <td>4</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>313</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td></td>
    </tr>
    <tr>
      <th>4</th>
      <td>b</td>
      <td>1</td>
      <td>3</td>
      <td>3</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>313</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Nonprofit</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 65 columns</p>
</div>




```python
df2= data[['race','call']]
print(df2)
```

         race  call
    0       w   0.0
    1       w   0.0
    2       b   0.0
    3       b   0.0
    4       w   0.0
    5       w   0.0
    6       w   0.0
    7       b   0.0
    8       b   0.0
    9       b   0.0
    10      b   0.0
    11      w   0.0
    12      b   0.0
    13      w   0.0
    14      b   0.0
    15      w   0.0
    16      w   0.0
    17      b   0.0
    18      w   0.0
    19      b   0.0
    20      b   0.0
    21      w   0.0
    22      w   0.0
    23      w   0.0
    24      w   0.0
    25      b   0.0
    26      b   0.0
    27      w   0.0
    28      b   0.0
    29      b   0.0
    ...   ...   ...
    4840    b   0.0
    4841    b   0.0
    4842    b   0.0
    4843    w   1.0
    4844    b   0.0
    4845    w   0.0
    4846    w   1.0
    4847    w   1.0
    4848    b   1.0
    4849    b   0.0
    4850    b   0.0
    4851    w   0.0
    4852    w   0.0
    4853    b   0.0
    4854    w   0.0
    4855    w   0.0
    4856    b   0.0
    4857    b   0.0
    4858    b   0.0
    4859    b   1.0
    4860    w   0.0
    4861    w   1.0
    4862    w   0.0
    4863    w   0.0
    4864    b   0.0
    4865    b   0.0
    4866    b   0.0
    4867    w   0.0
    4868    b   0.0
    4869    w   0.0
    
    [4870 rows x 2 columns]
    


```python
df2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 4870 entries, 0 to 4869
    Data columns (total 2 columns):
    race    4870 non-null object
    call    4870 non-null float32
    dtypes: float32(1), object(1)
    memory usage: 95.1+ KB
    

          A z test would be the test we would want to use here. We can load the appropriate columns that need to be examined, the **race** and **call** columns, into a new dataframe. This will be a two-sample test and we can determine the mean for both groups. The data set is large enough to make CLT apply. Our null-hythosis,**H0**, would be that there is no statistical significance between the number of interviews granted for white versus black applicants.. Our alternative hypothesis, **H1**, would be that there is a statistical significance between the number of interviews granted for white versus black applicants. 


```python
w = data[data.race=='w']
b = data[data.race=='b']
```


```python
# Call back rate of white and black sounding names
white= sum(df2.call[df2.race=='w'])/sum(df2.race=='w')
black= sum(df2.call[df2.race=='b'])/ sum(df2.race=='b')
```


```python
# Number of white and black sounding names
white_n= sum(df2.race=='w')
black_n= sum(df2.race=='b')
```


```python
# Callback probability difference and standard deviation
diff=white-black
samples= (white_n*white + black_n*black)/ (white_n + black_n)
s= np.sqrt(samples*(1-samples)/white_n) + (samples*(1-samples)/black_n)
```


```python
print(diff)
print(s)
```

    0.0320328542094
    0.00554363242347
    


```python
# Z score, p-value
z=(white-black-0)/s
p=(1-stats.norm.cdf(z))*2
print(z,p)
```

    5.77831496796 7.54524598356e-09
    

Looking at our p-value for our calculated z value, it seems that we do have a statistally significant result because our p value is in fact lower than out 0.05 alpha value. we would then say it is or **H1** or alternative hypothesis that our values support. We now need to calculate of 95% interval of our z value and our percentage error.


```python
z_critical=stats.norm.ppf(q=0.975)
error= (z_critical * s)
print(z_critical, error)
```

    1.95996398454 0.0108653198935
    


```python
max= diff + error
min= diff - error
print(max, min)
```

    0.042898174103 0.0211675343159
    

The margin for error was calculated at 0.011 and applying it to our difference in call back rates, it adjusts to a max value of 0.043 and a min value of 0.021 at the 95% confidence interval.


```python
replicates= np.empty(10000)
white= df2[df2.race=='w'].call.values
black= df2[df2.race=='b'].call.values
diff_mean= np.mean(white) - np.mean(black)

for i in range(len(replicates)):
    per_samples= np.random.permutation(np.concatenate((white, black)))
    white_perm = per_samples[:len(white)]
    black_perm = per_samples[len(black):]
    
    replicates[i]=np.abs(np.mean(white_perm) - np.mean(black_perm))
    
p=np.sum(replicates>diff_mean)/len(replicates)
print(p)
```

    0.0
    

Using bootstaps, we get a probability of 0 times in 10000 of getting a value of that is as great as our sample difference in means. This would indicate there could be a statistical difference in the number of white applicants vs black applicants that receive interviews.

   Our task was to use data on racial descrimination in the workplace. We examined if there was any significant
statistical difference between applicants that had white re is evidence that sounding names versus those with black sounding names. According to the P value from our z test and our bootstrap testing, we found that there appears to be a statistical difference between both groups and the number of interviews each received.
