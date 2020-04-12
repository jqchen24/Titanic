```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from scipy import stats          # For statistics

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

'''Plotly visualization .'''
import plotly.offline as py
from plotly.offline import iplot, init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected = True) # Required to use plotly offline in jupyter notebook
```


<script type="text/javascript">
window.PlotlyConfig = {MathJaxConfig: 'local'};
if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
if (typeof require !== 'undefined') {
require.undef("plotly");
requirejs.config({
    paths: {
        'plotly': ['https://cdn.plot.ly/plotly-latest.min']
    }
});
require(['plotly'], function(Plotly) {
    window._Plotly = Plotly;
});
}
</script>




```python
pwd
```




    '/home/jchen16/kaggle'




```python
train = pd.read_csv('~/kaggle/input/titanic/train.csv')
test = pd.read_csv('~/kaggle/input/titanic/test.csv')
```


```python
merged = pd.concat([train, test], sort = False)
```

- Merge train and test data together. This eliminates the hassle of handling train and test data seperately for various analysis.


```python
merged.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1309.000000</td>
      <td>891.000000</td>
      <td>1309.000000</td>
      <td>1046.000000</td>
      <td>1309.000000</td>
      <td>1309.000000</td>
      <td>1308.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>655.000000</td>
      <td>0.383838</td>
      <td>2.294882</td>
      <td>29.881138</td>
      <td>0.498854</td>
      <td>0.385027</td>
      <td>33.295479</td>
    </tr>
    <tr>
      <th>std</th>
      <td>378.020061</td>
      <td>0.486592</td>
      <td>0.837836</td>
      <td>14.413493</td>
      <td>1.041658</td>
      <td>0.865560</td>
      <td>51.758668</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.170000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>328.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>21.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.895800</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>655.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.454200</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>982.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>39.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>31.275000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1309.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>80.000000</td>
      <td>8.000000</td>
      <td>9.000000</td>
      <td>512.329200</td>
    </tr>
  </tbody>
</table>
</div>




```python
merged.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1309 entries, 0 to 417
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  1309 non-null   int64  
     1   Survived     891 non-null    float64
     2   Pclass       1309 non-null   int64  
     3   Name         1309 non-null   object 
     4   Sex          1309 non-null   object 
     5   Age          1046 non-null   float64
     6   SibSp        1309 non-null   int64  
     7   Parch        1309 non-null   int64  
     8   Ticket       1309 non-null   object 
     9   Fare         1308 non-null   float64
     10  Cabin        295 non-null    object 
     11  Embarked     1307 non-null   object 
    dtypes: float64(3), int64(4), object(5)
    memory usage: 132.9+ KB



```python
merged.shape
```




    (1309, 12)




```python
merged.columns
```




    Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
           'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],
          dtype='object')




```python
merged.dtypes
```




    PassengerId      int64
    Survived       float64
    Pclass           int64
    Name            object
    Sex             object
    Age            float64
    SibSp            int64
    Parch            int64
    Ticket          object
    Fare           float64
    Cabin           object
    Embarked        object
    dtype: object




```python
merged['Pclass'] = merged['Pclass'].astype('object')

merged['Survived_cat'] = pd.Categorical(merged['Survived'])
```


```python
merged
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Survived_cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1.0</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1.0</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>413</th>
      <td>1305</td>
      <td>NaN</td>
      <td>3</td>
      <td>Spector, Mr. Woolf</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>A.5. 3236</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>414</th>
      <td>1306</td>
      <td>NaN</td>
      <td>1</td>
      <td>Oliva y Ocana, Dona. Fermina</td>
      <td>female</td>
      <td>39.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17758</td>
      <td>108.9000</td>
      <td>C105</td>
      <td>C</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>415</th>
      <td>1307</td>
      <td>NaN</td>
      <td>3</td>
      <td>Saether, Mr. Simon Sivertsen</td>
      <td>male</td>
      <td>38.5</td>
      <td>0</td>
      <td>0</td>
      <td>SOTON/O.Q. 3101262</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>416</th>
      <td>1308</td>
      <td>NaN</td>
      <td>3</td>
      <td>Ware, Mr. Frederick</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>359309</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>417</th>
      <td>1309</td>
      <td>NaN</td>
      <td>3</td>
      <td>Peter, Master. Michael J</td>
      <td>male</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>2668</td>
      <td>22.3583</td>
      <td>NaN</td>
      <td>C</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>1309 rows × 13 columns</p>
</div>




```python
merged['Cabin'].value_counts()
```




    C23 C25 C27        6
    B57 B59 B63 B66    5
    G6                 5
    D                  4
    F4                 4
                      ..
    E63                1
    C128               1
    F                  1
    C103               1
    B4                 1
    Name: Cabin, Length: 186, dtype: int64



We don't really care about the frequency of each cabin; instead, we care about how many different kinds of cabin there are.


```python
merged['Cabin'].value_counts().count()
```




    186




```python
merged.isnull().sum()/merged.shape[0]
```




    PassengerId     0.000000
    Survived        0.319328
    Pclass          0.000000
    Name            0.000000
    Sex             0.000000
    Age             0.200917
    SibSp           0.000000
    Parch           0.000000
    Ticket          0.000000
    Fare            0.000764
    Cabin           0.774637
    Embarked        0.001528
    Survived_cat    0.319328
    dtype: float64




```python
merged.groupby('Sex').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>648.186695</td>
      <td>0.742038</td>
      <td>28.687088</td>
      <td>0.652361</td>
      <td>0.633047</td>
      <td>46.198097</td>
    </tr>
    <tr>
      <th>male</th>
      <td>658.766311</td>
      <td>0.188908</td>
      <td>30.585228</td>
      <td>0.413998</td>
      <td>0.247924</td>
      <td>26.154601</td>
    </tr>
  </tbody>
</table>
</div>




```python
merged.groupby('Sex')['Survived'].mean()
```




    Sex
    female    0.742038
    male      0.188908
    Name: Survived, dtype: float64




```python
merged['Sex'].value_counts()
```




    male      843
    female    466
    Name: Sex, dtype: int64




```python
merged['Sex'].value_counts(normalize = True)
```




    male      0.644003
    female    0.355997
    Name: Sex, dtype: float64




```python
merged['Survived'].value_counts(normalize = True)
```




    0.0    0.616162
    1.0    0.383838
    Name: Survived, dtype: float64



### Univariate charts


```python
sns.catplot(x = 'Sex', kind = 'count', data = merged)
```




    <seaborn.axisgrid.FacetGrid at 0x7f9bf094d1d0>




![png](output_22_1.png)



```python
sns.catplot(x = 'SibSp', kind = 'count', data = merged)
```




    <seaborn.axisgrid.FacetGrid at 0x7f9bf094d7b8>




![png](output_23_1.png)



```python
p1=sns.kdeplot(merged['Fare'], shade=True, color="r")
p1=sns.kdeplot(merged['Age'], shade=True, color="b")


```


![png](output_24_0.png)



```python
merged['Sex'].value_counts().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9bf0794278>




![png](output_25_1.png)



```python
merged['Sex'].value_counts(normalize = True).plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9bf070d550>




![png](output_26_1.png)


Create a histogram and density plot


```python
sns.distplot(merged['Age'].dropna())

```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9bf06e4c88>




![png](output_28_1.png)



```python
sns.distplot(merged['Age'].dropna(), kde = False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9bf067d320>




![png](output_29_1.png)



```python
sns.distplot(merged['Age'].dropna(), color = 'blue', axlabel = 'Age distribution')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9bf0630ba8>




![png](output_30_1.png)



```python
sns.distplot(merged['Fare'].dropna(), axlabel = 'Fare distribution')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9bf0548b38>




![png](output_31_1.png)


Another way to create a histogram


```python
merged['Age'].hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9bf04178d0>




![png](output_33_1.png)



```python
sns.distplot(merged['SibSp'])


```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9bf067cc88>




![png](output_34_1.png)


The following works too, although it's designed for categorical variables. 


```python
sns.catplot(x = 'SibSp', kind = 'count', data = merged)
```




    <seaborn.axisgrid.FacetGrid at 0x7f9bf0361550>




![png](output_36_1.png)



```python
sns.countplot(x = 'Parch', data = merged)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9bf02c7550>




![png](output_37_1.png)


### bi-variate charts


```python
sns.catplot(x = 'Survived', y = 'Age', kind = 'box', data = merged)
sns.catplot(x = 'Survived', y = 'Fare', kind = 'box', data = merged)
```




    <seaborn.axisgrid.FacetGrid at 0x7f9bf02600b8>




![png](output_39_1.png)



![png](output_39_2.png)



```python
sns.catplot(y="Survived", x = 'Sex', data = merged, kind = 'bar')
# Note the following code doesn't work as countplot only works for one variable.
# sns.catplot(y = 'Survived', x = 'Sex', data = merged, kind = 'count')
```




    <seaborn.axisgrid.FacetGrid at 0x7f9bf02395c0>




![png](output_40_1.png)



```python
sns.catplot(x="Sex", y="Survived", hue="Pclass", kind="bar", data=merged);


```


![png](output_41_0.png)



```python
# Following code doesn't work because countplot only works for single variable!!!!!
# sns.countplot(x = 'Age', y = 'Survived', data = train)
sns.catplot(x = 'Age', y = 'Survived', kind = 'bar', data = merged)
```




    <seaborn.axisgrid.FacetGrid at 0x7f9bf0483b38>




![png](output_42_1.png)


The following chart that uses categorical version of Survived doesn't make sense


```python
sns.catplot(x = 'Age', y = 'Survived_cat', kind = 'bar', data = merged)
```




    <seaborn.axisgrid.FacetGrid at 0x7f9bf0203898>




![png](output_44_1.png)



```python
sns.catplot(x = 'Survived', y = 'Age', kind = 'bar', data = merged)
```




    <seaborn.axisgrid.FacetGrid at 0x7f9befd2b9e8>




![png](output_45_1.png)



```python
sns.catplot(x = 'Pclass', y = 'Survived', data = merged, kind = 'bar')
```




    <seaborn.axisgrid.FacetGrid at 0x7f9bf07a0978>




![png](output_46_1.png)


The following chart doesn't make sense.


```python
sns.catplot(x = 'Pclass', y = 'Survived_cat', data = merged, kind = 'bar')
```




    <seaborn.axisgrid.FacetGrid at 0x7f9befd24eb8>




![png](output_48_1.png)



```python
sns.catplot(y = 'Age', x = 'Survived', data = merged, kind = 'bar')
```




    <seaborn.axisgrid.FacetGrid at 0x7f9befc7c5c0>




![png](output_49_1.png)



```python
sns.catplot(x = 'SibSp', y = 'Survived', data = merged, kind = 'bar')
```




    <seaborn.axisgrid.FacetGrid at 0x7f9befc43fd0>




![png](output_50_1.png)


Equivalent to barplot function


```python
sns.barplot(x = 'SibSp', y = 'Survived', data = merged)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9befad1278>




![png](output_52_1.png)



```python
sns.catplot(x = 'Survived', y = 'SibSp', data = merged, kind = 'bar')
```




    <seaborn.axisgrid.FacetGrid at 0x7f9befbaca20>




![png](output_53_1.png)



```python
sns.catplot(x = 'Parch', y = 'Survived', data = merged, kind = 'bar')
```




    <seaborn.axisgrid.FacetGrid at 0x7f9befab92b0>




![png](output_54_1.png)



```python
sns.catplot(x = 'Pclass', y = 'Survived', data = merged, kind = 'bar')
```




    <seaborn.axisgrid.FacetGrid at 0x7f9bef9c7b70>




![png](output_55_1.png)



```python
sns.catplot(x = 'Sex', y = 'Survived', hue = 'Pclass', data = merged, kind = 'bar')
```




    <seaborn.axisgrid.FacetGrid at 0x7f9bef91e208>




![png](output_56_1.png)


#### Break out the chart into different sections by a categorical variable


```python
sns.catplot(x = 'Sex', y = 'Survived', col = 'Pclass', data = merged, kind = 'bar')
```




    <seaborn.axisgrid.FacetGrid at 0x7f9bf0aa8550>




![png](output_58_1.png)


Although in a countplot function, there cannot have both x and y, we can use hue.


```python
sns.countplot(x = 'Sex', hue = 'Survived', data = merged)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9bef884ef0>




![png](output_60_1.png)



```python
sns.countplot(x = 'SibSp', hue = 'Survived', data = merged)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9bef7e1ba8>




![png](output_61_1.png)



```python
sns.countplot(x = 'SibSp', hue = 'Survived', data = merged)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9bef775898>




![png](output_62_1.png)



```python
# Have to use categorical variable here; otherwise the chart will not show up.
sns.catplot(x="Fare",y="Survived_cat",kind='violin',data=merged)
sns.catplot(x="Age",y="Survived_cat",kind='violin',data=merged)
sns.catplot(x="SibSp",y="Survived_cat",kind='violin',data=merged)
```




    <seaborn.axisgrid.FacetGrid at 0x7f9bef7b6c88>




![png](output_63_1.png)



![png](output_63_2.png)



![png](output_63_3.png)


### Line plot - numerical vs. numerical variables.


```python
sns.lineplot(x = 'Age', y = 'SibSp', data = merged)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9bef567160>




![png](output_65_1.png)



```python
sns.lineplot(x = 'Age', y = 'SibSp', hue = 'Pclass', data = merged)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9bef4dd128>




![png](output_66_1.png)



```python
merged.groupby(['Sex'])['Survived'].mean().plot(kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9bef46c588>




![png](output_67_1.png)



```python
merged.groupby(['Survived'])['Age'].mean().plot(kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9bef3e6cf8>




![png](output_68_1.png)



```python
merged.groupby(['Survived'])['SibSp'].mean().plot(kind = 'bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9bef3c0b00>




![png](output_69_1.png)



```python
merged.groupby(['Survived'])['Parch'].mean().plot(kind = 'bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9bef331470>




![png](output_70_1.png)



```python
pd.crosstab(merged['Sex'], merged['Pclass'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Pclass</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>144</td>
      <td>106</td>
      <td>216</td>
    </tr>
    <tr>
      <th>male</th>
      <td>179</td>
      <td>171</td>
      <td>493</td>
    </tr>
  </tbody>
</table>
</div>



## Feature engineering

### Cabin


```python
merged['Cabin'].value_counts().count()
```




    186




```python
merged['Cabin'].value_counts(dropna = False).count()
```




    187



It is reasonable to presume that those NaNs didn't have a cabin, which could tell us something about 'Survived'. We will flag NaN as 'X' and keep only the 1st character where Cabin has alphanumeric values.


```python
"""Flag all the NaNs of Cabin as 'X'."""
merged['Cabin'].fillna(value = 'X', inplace = True)
merged['Cabin'].isnull().sum()
```




    0




```python
'''Keep only the 1st character where Cabin is alphanumerical.'''
merged['Cabin'] = merged['Cabin'].apply( lambda x : x[0])
display(merged['Cabin'].value_counts())
```


    X    1014
    C      94
    B      65
    D      46
    E      41
    A      22
    F      21
    G       5
    T       1
    Name: Cabin, dtype: int64



```python
sns.catplot(x = 'Cabin', data = merged, kind = 'count')
```




    <seaborn.axisgrid.FacetGrid at 0x7f9bef352cc0>




![png](output_79_1.png)



```python
pd.options.display.max_rows = 100
```

### Name


```python
display(merged['Name'])
```


    0                                Braund, Mr. Owen Harris
    1      Cumings, Mrs. John Bradley (Florence Briggs Th...
    2                                 Heikkinen, Miss. Laina
    3           Futrelle, Mrs. Jacques Heath (Lily May Peel)
    4                               Allen, Mr. William Henry
                                 ...                        
    413                                   Spector, Mr. Woolf
    414                         Oliva y Ocana, Dona. Fermina
    415                         Saether, Mr. Simon Sivertsen
    416                                  Ware, Mr. Frederick
    417                             Peter, Master. Michael J
    Name: Name, Length: 1309, dtype: object



```python
merged['Title'] = merged['Name'].str.extract('([A-Za-z]+)\.')
merged['Title'].value_counts()
```




    Mr          757
    Miss        260
    Mrs         197
    Master       61
    Dr            8
    Rev           8
    Col           4
    Mlle          2
    Ms            2
    Major         2
    Sir           1
    Capt          1
    Lady          1
    Countess      1
    Jonkheer      1
    Dona          1
    Don           1
    Mme           1
    Name: Title, dtype: int64



We can see there are several titles with the very least frequency. So, it makes sense to put them in fewer buckets. Professionals like Dr, Rev, Col, Major, Capt will be put into 'Officer' bucket. Titles such as Dona, Jonkheer, Countess, Sir, Lady, Don were usually entitled to the aristocrats and hence these titles will be put into bucket 'Aristocrat'. We would also replace Mlle and Ms with Miss and Mme by Mrs as these are French titles.


```python
'''Create a bucket Officer and put Dr, Rev, Col, Major, Capt titles into it.'''
merged['Title'].replace(to_replace = ['Dr', 'Rev', 'Col', 'Major', 'Capt'], value = 'Officer', inplace = True)

'''Put Dona, Jonkheer, Countess, Sir, Lady, Don in bucket Aristocrat.'''
merged['Title'].replace(to_replace = ['Dona', 'Jonkheer', 'Countess', 'Sir', 'Lady', 'Don'], value = 'Aristocrat', inplace = True)

'''Finally Replace Mlle and Ms with Miss. And Mme with Mrs.'''
merged['Title'].replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'}, inplace = True)

merged['Title'].value_counts()
```




    Mr            757
    Miss          264
    Mrs           198
    Master         61
    Officer        23
    Aristocrat      6
    Name: Title, dtype: int64



### Family size

Create a new variable 'Family_size' from SibSp & Parch


```python
merged['Family_size'] = merged['SibSp'] + merged['Parch'] + 1
merged['Family_size'].value_counts().plot.bar()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9bef24b128>




![png](output_88_1.png)


We will create 4 buckets namely single, small, medium, and large for rest of them.


```python
'''Create buckets of single, small, medium, and large and then put respective values into them.'''
merged['Family_size'].replace(to_replace = [1], value = 'single', inplace = True)
merged['Family_size'].replace(to_replace = [2,3], value = 'small', inplace = True)
merged['Family_size'].replace(to_replace = [4,5], value = 'medium', inplace = True)
merged['Family_size'].replace(to_replace = np.arange(6,12), value = 'large', inplace = True)

merged['Family_size'].value_counts()
```




    single    790
    small     394
    medium     65
    large      60
    Name: Family_size, dtype: int64



### Ticket


```python
merged['Ticket'].value_counts()
```




    CA. 2343    11
    CA 2144      8
    1601         8
    PC 17608     7
    347082       7
                ..
    364499       1
    348121       1
    345768       1
    350408       1
    65306        1
    Name: Ticket, Length: 929, dtype: int64



We will use IQR method to detect the outliers for variable Age and Fare though we won't remove them.


```python
print("the 50% quartile of age is {}".format(merged['Age'].quantile(0.5)))
```

    the 50% quartile of age is 28.0



```python
'''Create a function to count total outliers. And plot variables with and without outliers.'''
def outliers(variable):
    global filtered
    # Calculate 1st, 3rd quartiles and iqr.
    q1, q3 = variable.quantile(0.25), variable.quantile(0.75)
    iqr = q3 - q1
    
    # Calculate lower fence and upper fence for outliers
    l_fence, u_fence = q1 - 1.5*iqr , q3 + 1.5*iqr   # Any values less than l_fence and greater than u_fence are outliers.
    
    # Observations that are outliers
    outliers = variable[(variable<l_fence) | (variable>u_fence)]
    print('Total Outliers of', variable.name,':', outliers.count())
    
    # Drop obsevations that are outliers
    filtered = variable.drop(outliers.index, axis = 0)

    # Create subplots
    out_variables = [variable, filtered]
    out_titles = [' Distribution with Outliers', ' Distribution Without Outliers']
    title_size = 25
    font_size = 18
    plt.figure(figsize = (25, 15))
    for ax, outlier, title in zip(range(1,3), out_variables, out_titles):
        plt.subplot(2, 1, ax)
        sns.boxplot(outlier).set_title('%s' %outlier.name + title, fontsize = title_size)
        plt.xticks(fontsize = font_size)
        plt.xlabel('%s' %outlier.name, fontsize = font_size)
```


```python
'''Count total outliers of Age. Plot Age with and without outliers.'''
outliers(merged.Age)
```

    Total Outliers of Age : 9



![png](output_96_1.png)


### Impute missing values


```python
merged.isnull().sum() > 0
```




    PassengerId     False
    Survived         True
    Pclass          False
    Name            False
    Sex             False
    Age              True
    SibSp           False
    Parch           False
    Ticket          False
    Fare             True
    Cabin           False
    Embarked         True
    Survived_cat     True
    Title           False
    Family_size     False
    dtype: bool



For categorical variables mode-imputation is performed and for numerical variable mean-impuation is performed **if its distribution is symmetric** (or almost symmetric or normal like Age). On the other hand, for a variable with skewed distribution and outliers (like Fare), meadian-imputation is recommended as median is more immune to outliers than mean.

However, one clear disadvantage of using mean, median or mode to impute missing values is the addition of bias if the amount of missing values is significant (like Age). So simply replacing them with the mean or the median age might not be the best solution since the age may differ by groups and categories of passengers.

To solve this, we can group our data by some variables that have no missing values and for each subset compute the median age to impute the missing values. Or we can build a linear regression model that will predict missing values of Age using the features that have no missing values. These two methods may result in better accuracy without high bias, unless a missing value is expected to have a very high variance. We will show the former method of imputation

*Let's first look for variables with the missing values using Plotly's scatter plot.*


```python
'''Function to plot scatter plot'''
def scatter_plot(x, y, title, yaxis, size, c_scale):
    trace = go.Scatter(
    x = x,
    y = y,
    mode = 'markers',
    marker = dict(color = y, size = size, showscale = True, colorscale = c_scale))
    layout = go.Layout(hovermode= 'closest', title = title, yaxis = dict(title = yaxis))
    fig = go.Figure(data = [trace], layout = layout)
    return iplot(fig)
```

#### Nice trick to select the columns with missing values


```python
merged.loc[:, merged.isnull().sum()>0].count()
```




    Survived         891
    Age             1046
    Fare            1308
    Embarked        1307
    Survived_cat     891
    dtype: int64




```python
merged['Fare'].count()
```




    1308




```python
missing_columns = len(merged) - merged.loc[:, merged.isnull().sum()>0].count()
missing_columns
# shows how many missing values there are for the columns
```




    Survived        418
    Age             263
    Fare              1
    Embarked          2
    Survived_cat    418
    dtype: int64




```python
x = missing_columns.index
x
y = missing_columns
y
```




    Survived        418
    Age             263
    Fare              1
    Embarked          2
    Survived_cat    418
    dtype: int64




```python
'''Get and plot only the missing columns with their missing values.'''
missing_columns = len(merged) - merged.loc[:, merged.isnull().sum()>0].count()
x = missing_columns.index
y = missing_columns
title = 'Features with Missing Values'
scatter_plot(x, y, title, 'Missing Values', 30, 'Rainbow')
```


<div>


            <div id="5b4aafc1-1eb7-40e0-ba2e-912719e7d6bc" class="plotly-graph-div" style="height:525px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};

                if (document.getElementById("5b4aafc1-1eb7-40e0-ba2e-912719e7d6bc")) {
                    Plotly.newPlot(
                        '5b4aafc1-1eb7-40e0-ba2e-912719e7d6bc',
                        [{"marker": {"color": [418, 263, 1, 2, 418], "colorscale": [[0.0, "rgb(150,0,90)"], [0.125, "rgb(0,0,200)"], [0.25, "rgb(0,25,255)"], [0.375, "rgb(0,152,255)"], [0.5, "rgb(44,255,150)"], [0.625, "rgb(151,255,0)"], [0.75, "rgb(255,234,0)"], [0.875, "rgb(255,111,0)"], [1.0, "rgb(255,0,0)"]], "showscale": true, "size": 30}, "mode": "markers", "type": "scatter", "x": ["Survived", "Age", "Fare", "Embarked", "Survived_cat"], "y": [418, 263, 1, 2, 418]}],
                        {"hovermode": "closest", "template": {"data": {"bar": [{"error_x": {"color": "#2a3f5f"}, "error_y": {"color": "#2a3f5f"}, "marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "bar"}], "barpolar": [{"marker": {"line": {"color": "#E5ECF6", "width": 0.5}}, "type": "barpolar"}], "carpet": [{"aaxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "baxis": {"endlinecolor": "#2a3f5f", "gridcolor": "white", "linecolor": "white", "minorgridcolor": "white", "startlinecolor": "#2a3f5f"}, "type": "carpet"}], "choropleth": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "choropleth"}], "contour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "contour"}], "contourcarpet": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "contourcarpet"}], "heatmap": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmap"}], "heatmapgl": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "heatmapgl"}], "histogram": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "histogram"}], "histogram2d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2d"}], "histogram2dcontour": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "histogram2dcontour"}], "mesh3d": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "type": "mesh3d"}], "parcoords": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "parcoords"}], "pie": [{"automargin": true, "type": "pie"}], "scatter": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter"}], "scatter3d": [{"line": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatter3d"}], "scattercarpet": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattercarpet"}], "scattergeo": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergeo"}], "scattergl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattergl"}], "scattermapbox": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scattermapbox"}], "scatterpolar": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolar"}], "scatterpolargl": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterpolargl"}], "scatterternary": [{"marker": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "type": "scatterternary"}], "surface": [{"colorbar": {"outlinewidth": 0, "ticks": ""}, "colorscale": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "type": "surface"}], "table": [{"cells": {"fill": {"color": "#EBF0F8"}, "line": {"color": "white"}}, "header": {"fill": {"color": "#C8D4E3"}, "line": {"color": "white"}}, "type": "table"}]}, "layout": {"annotationdefaults": {"arrowcolor": "#2a3f5f", "arrowhead": 0, "arrowwidth": 1}, "coloraxis": {"colorbar": {"outlinewidth": 0, "ticks": ""}}, "colorscale": {"diverging": [[0, "#8e0152"], [0.1, "#c51b7d"], [0.2, "#de77ae"], [0.3, "#f1b6da"], [0.4, "#fde0ef"], [0.5, "#f7f7f7"], [0.6, "#e6f5d0"], [0.7, "#b8e186"], [0.8, "#7fbc41"], [0.9, "#4d9221"], [1, "#276419"]], "sequential": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]], "sequentialminus": [[0.0, "#0d0887"], [0.1111111111111111, "#46039f"], [0.2222222222222222, "#7201a8"], [0.3333333333333333, "#9c179e"], [0.4444444444444444, "#bd3786"], [0.5555555555555556, "#d8576b"], [0.6666666666666666, "#ed7953"], [0.7777777777777778, "#fb9f3a"], [0.8888888888888888, "#fdca26"], [1.0, "#f0f921"]]}, "colorway": ["#636efa", "#EF553B", "#00cc96", "#ab63fa", "#FFA15A", "#19d3f3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"], "font": {"color": "#2a3f5f"}, "geo": {"bgcolor": "white", "lakecolor": "white", "landcolor": "#E5ECF6", "showlakes": true, "showland": true, "subunitcolor": "white"}, "hoverlabel": {"align": "left"}, "hovermode": "closest", "mapbox": {"style": "light"}, "paper_bgcolor": "white", "plot_bgcolor": "#E5ECF6", "polar": {"angularaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "radialaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "scene": {"xaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "yaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}, "zaxis": {"backgroundcolor": "#E5ECF6", "gridcolor": "white", "gridwidth": 2, "linecolor": "white", "showbackground": true, "ticks": "", "zerolinecolor": "white"}}, "shapedefaults": {"line": {"color": "#2a3f5f"}}, "ternary": {"aaxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "baxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}, "bgcolor": "#E5ECF6", "caxis": {"gridcolor": "white", "linecolor": "white", "ticks": ""}}, "title": {"x": 0.05}, "xaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}, "yaxis": {"automargin": true, "gridcolor": "white", "linecolor": "white", "ticks": "", "title": {"standoff": 15}, "zerolinecolor": "white", "zerolinewidth": 2}}}, "title": {"text": "Features with Missing Values"}, "yaxis": {"title": {"text": "Missing Values"}}},
                        {"responsive": true}
                    ).then(function(){

var gd = document.getElementById('5b4aafc1-1eb7-40e0-ba2e-912719e7d6bc');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


### Impute missing values for embarked and Fare


```python
sns.distplot(merged['Fare'].dropna())
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f9bef106588>




![png](output_108_1.png)



```python
merged['Embarked'].value_counts()
```




    S    914
    C    270
    Q    123
    Name: Embarked, dtype: int64




```python
'''Impute missing values of Embarked. Embarked is a categorical variable where S is the most frequent.'''
merged.Embarked.fillna(value = 'S', inplace = True)

'''Impute missing values of Fare. Fare is a numerical variable with outliers. Hence it will be imputed by median.'''
merged.Fare.fillna(value = merged.Fare.median(), inplace = True)
```

#### Impute age

To impute Age with grouped median, we need to know which features are heavily correlated with Age. Let's find out the variables correlated with Age.


```python
correlation = merged.loc[:, ['Sex', 'Pclass', 'Embarked', 'Title', 'Family_size', 'Parch', 'SibSp', 'Cabin', 'Ticket']]
fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize = (25,25))
sns.boxplot(x = correlation['Sex'], y =  merged.Age, ax = ax)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-120-9ccf089ade44> in <module>
          1 correlation = merged.loc[:, ['Sex', 'Pclass', 'Embarked', 'Title', 'Family_size', 'Parch', 'SibSp', 'Cabin', 'Ticket']]
          2 fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize = (25,25))
    ----> 3 sns.boxplot(x = correlation['Sex'], y =  merged.Age, ax = ax)
    

    NameError: name 'ax' is not defined



![png](output_113_1.png)



```python
## Zip is used to join two tuples together
zip(axes.flatten(), correlation.columns)
```




    <zip at 0x7f9bedc064c8>




```python
"""Create a boxplot to view the variables correlated with Age. First extract the variables we're interested in."""
correlation = merged.loc[:, ['Sex', 'Pclass', 'Embarked', 'Title', 'Family_size', 'Parch', 'SibSp', 'Cabin', 'Ticket']]
fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize = (25,25))
for ax, column in zip(axes.flatten(), correlation.columns):
    sns.boxplot(x = correlation[column], y =  merged.Age, ax = ax)
    ax.set_title(column, fontsize = 23)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 20)
    ax.set_ylabel('Age', fontsize = 20)
    ax.set_xlabel('')
fig.suptitle('Variables Associated with Age', fontsize = 30)
fig.tight_layout(rect = [0, 0.03, 1, 0.95])
```


![png](output_115_0.png)


**Findings**

- Age seems to be correlated with PClass, Title, Family_size, SibSp, Parch, and Cabin


```python
"""Let's plot correlation heatmap to see which variable is highly correlated with Age and if our boxplot interpretation holds true. 
We need to convert categorical variable into numerical to plot correlation heatmap. So convert categorical variables into numerical."""
from sklearn.preprocessing import LabelEncoder
correlation = correlation.agg(LabelEncoder().fit_transform)
correlation['Age'] = merged.Age # Inserting Age in variable correlation.
correlation = correlation.set_index('Age').reset_index() # Move Age at index 0.
```


```python
correlation['Family_size'].value_counts()
```




    2    790
    3    394
    1     65
    0     60
    Name: Family_size, dtype: int64




```python
# Now create the heatmap correlation
plt.figure(figsize=(15,6))
sns.heatmap(correlation.corr(), cmap ='BrBG',annot = True)
plt.title('Variables correlated with Age')
plt.show()
```


![png](output_119_0.png)


**Findings**: 

As expected Sex, Embarked, and Ticket have the weakest correlation with Age what we could guess beforehand from boxplot. Parch and Family_size are moderately correlated with Age. Title, Pclass, Cabin, and SibSp have the highest correlation with Age. But we are gonna use Title and Pclass only in order to impute Age since they have the strongest correlation with Age. So the tactic is to impute missing values of Age with the median age of similar rows according to Title and Pclass.


```python
merged.groupby(['Title', 'Pclass'])['Age'].mean()
```




    Title       Pclass
    Aristocrat  1         41.166667
    Master      1          6.984000
                2          2.757273
                3          6.090000
    Miss        1         30.131148
                2         20.865714
                3         17.360874
    Mr          1         41.450758
                2         32.346715
                3         28.318910
    Mrs         1         42.926471
                2         33.518519
                3         32.326531
    Officer     1         50.916667
                2         40.700000
    Name: Age, dtype: float64



**Note the use of transform along with lambda**


```python
'''Impute Age with median of respective columns (i.e., Title and Pclass).'''
merged.Age = merged.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
```


```python
merged.isnull().sum()
```




    PassengerId       0
    Survived        418
    Pclass            0
    Name              0
    Sex               0
    Age               0
    SibSp             0
    Parch             0
    Ticket            0
    Fare              0
    Cabin             0
    Embarked          0
    Survived_cat    418
    Title             0
    Family_size       0
    dtype: int64



If we want to normalize the variables, we can use the following code --

**df.transform(lambda x: (x - x.mean()) / x.std())**

### Bivariate analysis

Being the most important part, bivariate analysis tries to find the relationship between two variables. We will look for correlation or association between our predictor and target variables. Bivariate analysis is performed for any combination of categorical and numerical variables. The combination can be: Numerical & Numerical, Numerical & Categorical and Categorical & Categorical. Different methods are used to tackle these combinations during analysis process. The methods are:

- Numerical & Numerical: Pearson's correlation, or Spearman correlation (doesn't require normal distribution).
- Numerical & Categorical: Point biserial correlation (only if categorical variable is binary type), or ANOVA test. For this problem, you can use either biserial correlation or ANOVA. But I will perform both test just to learn because ANOVA will come in handy if categorical variable has more than two groups.
- Categorical & Categorical: We would use Chi-square test for bivariate analysis between categorical variables.

#### Numerical and categorical

First we create a boxplot between our numerical and categorical variables to check if the distribution of numerical variable is distinct in different classes of nominal variables. Then we find the mean of numerical variable for every class of categorical variable. Again we plot a histogram of numerical variable for every class of categorical variable. Finally anova or point biserial correlation (in case of two class categorical variable) is calculated to find association between nominal and numerical variables.


```python
"""Let's split the train and test data for bivariate analysis since test data has no Survived values. We need our target variable without missing values to conduct the association test with predictor variables."""
df_train = merged.iloc[:891, :]
df_test = merged.iloc[891:, :]
```


```python
'''#1.Create a function that creates boxplot between categorical and numerical variables and calculates biserial correlation.'''
def boxplot_and_correlation(cat,num):
    '''cat = categorical variable, and num = numerical variable.'''
    plt.figure(figsize = (18,7))
    title_size = 18
    font_size = 15
    ax = sns.boxplot(x = cat, y = num)
    
    # Select boxes to change the color
    box = ax.artists[0]
    box1 = ax.artists[1]
    
    # Change the appearance of that box
    box.set_facecolor('red')
    box1.set_facecolor('green')
    plt.title('Association between Survived & %s' %num.name, fontsize = title_size)
    plt.xlabel('%s' %cat.name, fontsize = font_size)
    plt.ylabel('%s' %num.name, fontsize = font_size)
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
    plt.show()
    print('Correlation between', num.name, 'and', cat.name,':', stats.pointbiserialr(num, cat))

'''#2.Create another function to calculate mean when grouped by categorical variable. And also plot the grouped mean.'''
def nume_grouped_by_cat(num, cat):
    global ax
    font_size = 15
    title_size = 18
    grouped_by_cat = num.groupby(cat).mean().sort_values( ascending = False)
    grouped_by_cat.rename ({1:'survived', 0:'died'}, axis = 'rows', inplace = True) # Renaming index
    grouped_by_cat = round(grouped_by_cat, 2)
    ax = grouped_by_cat.plot.bar(figsize = (18,5)) 
    abs_bar_labels()
    plt.title('Mean %s ' %num.name + ' of Survivors vs Victims', fontsize = title_size)
    plt.ylabel('Mean ' + '%s' %num.name, fontsize = font_size)
    plt.xlabel('%s' %cat.name, fontsize = font_size)
    plt.xticks(fontsize = font_size)
    plt.yticks(fontsize = font_size)
    plt.show()

'''#3.This function plots histogram of numerical variable for every class of categorical variable.'''
def num_hist_by_cat(num,cat):
    font_size = 15
    title_size = 18
    plt.figure(figsize = (18,7))
    num[cat == 1].hist(color = ['g'], label = 'Survived', grid = False)
    num[cat == 0].hist(color = ['r'], label = 'Died', grid = False)
    plt.yticks([])
    plt.xticks(fontsize = font_size)
    plt.xlabel('%s' %num.name, fontsize = font_size)
    plt.title('%s ' %num.name + ' Distribution of Survivors vs Victims', fontsize = title_size)
    plt.legend()
    plt.show()
   
'''#4.Create a function to calculate anova between numerical and categorical variable.'''
def anova(num, cat):
    from scipy import stats
    grp_num_by_cat_1 = num[cat == 1] # Group our numerical variable by categorical variable(1). Group Fair by survivors
    grp_num_by_cat_0 = num[cat == 0] # Group our numerical variable by categorical variable(0). Group Fare by victims
    f_val, p_val = stats.f_oneway(grp_num_by_cat_1, grp_num_by_cat_0) # Calculate f statistics and p value
    print('Anova Result between ' + num.name, ' & '+ cat.name, ':' , f_val, p_val)  
```

#### Fare and suvived


```python
boxplot_and_correlation(df_train.Survived, df_train.Fare)
```


```python
sns.catplot(x = 'Survived', y = 'Fare', data = df_train, kind = 'bar')
```


```python
df_train.loc[df_train['Survived'] == 1, 'Fare'].hist()
df_train.loc[df_train['Survived'] == 0, 'Fare'].hist()
```


```python
"""Let's perform ANOVA between Fare and Survived. One can omit this step. I perform just to show how anova is performed if there were more than two groups in our categorical variable."""
anova(df_train.Fare, df_train.Survived)
```

#### Age and Survived


```python
"""Let's create a box plot between Age and Survived to have an idea by how much Age is associated with Survived. Also find point biserial correlation between them."""
boxplot_and_correlation(df_train.Survived, df_train.Age)
```


```python
sns.catplot(x = 'Survived', y = 'Age', data = df_train, kind = 'bar')
```


```python
df_train.loc[df_train['Survived'] == 1, 'Age'].hist()
df_train.loc[df_train['Survived'] == 0, 'Age'].hist()
```


```python
'''Perform ANOVA between all the levels of Survived (i.e.., 0 and 1) and Age.'''
anova(df_train.Age, df_train.Survived)
```


```python

```
