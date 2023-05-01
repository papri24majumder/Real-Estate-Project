import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import matplotlib
matplotlib.rcParams['figure.figsize'] = (20,20)

df = pd.read_csv('Bengaluru_House_Data.csv')
df.head()

df.shape

df.groupby('area_type')['area_type'].agg('count')

df1=df.drop(['area_type','availability','society','balcony'], axis='columns')
df1.head()

df1.isnull().sum()

df2=df1.dropna()
df2.isnull().sum()

df2.shape

df2['size'].unique()

df2['BHK']=df2['size'].apply(lambda x: int(x.split(' ')[0]))

df2.head()

df2['BHK'].unique()

def is_float(x):
    try:
        float(x)
    except:
        return False
    return True

df2[~df2['total_sqft'].apply(is_float)].head(10)

def convert(x):
    tokens = x.split('-')
    if(len(tokens)==2):
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None

df3=df2.copy()
df3['total_sqft'] = df3['total_sqft'].apply(convert)
df3.head()

df4=df3.copy()
df4['price_per_sqft'] = df4['price']*100000/df4['total_sqft']
df4.head()

len(df4.location.unique())

df4.location = df4.location.apply(lambda x:x.strip())

location_stats = df4.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats

len(location_stats[location_stats<=10])

location_stats_less_than_10 = location_stats[location_stats<=10]
location_stats_less_than_10

df4.location = df4.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)

len(df4.location.unique())

df4[df4.total_sqft/df4.BHK<300].head()

df5=df4[~(df4.total_sqft/df4.BHK<300)]
df5.shape

df5.price_per_sqft.describe()

def remove_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df6=remove_outliers(df5)
df6.shape

def plot_scatter_chart(df, location):
    bhk2 = df[(df.location==location) & (df.BHK==2)]
    bhk3 = df[(df.location==location) & (df.BHK==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue', label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+',color='green', label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price")
    plt.title(location)
    plt.legend()
   
plot_scatter_chart(df6, "Rajaji Nagar")

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('BHK'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('BHK'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df7 = remove_bhk_outliers(df6)
# df8 = df7.copy()
df7.shape

plot_scatter_chart(df7, "Rajaji Nagar")

import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,10)
plt.hist(df7.price_per_sqft,rwidth=0.8)
plt.xlabel("Price per square Feet")
plt.ylabel("Count")

df7[df7.bath>df7.BHK+2]

df8=df7[df7.bath<df7.BHK+2]
df8.shape

df9=df8.drop(['size','price_per_sqft'],axis='columns')
df9.head(3)

dummies = pd.get_dummies(df9.location)
dummies.head()

df10 = pd.concat([df9,dummies.drop('other', axis='columns')], axis='columns')
df10.head()

df11 = df10.drop('location', axis='columns')
df11.head()

df11.shape

x=df11.drop('price',axis='columns')
x.head()

y = df11.price
y.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=10)

from sklearn.linear_model import LinearRegression
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), x, y, cv=cv)

x.columns

def predict_price(location,sqft,bath,BHK):    
    loc_index = np.where(x.columns==location)[0][0]

    x1 = np.zeros(len(x.columns))
    x1[0] = sqft
    x1[1] = bath
    x1[2] = BHK
    if loc_index >= 0:
        x1[loc_index] = 1

    return lr_clf.predict([x1])[0]

predict_price('1st Phase JP Nagar',1000, 3, 3)

predict_price('Yelenahalli',2000, 3, 3)

import pickle
with open('Bengaluru_House_Data.pickle','wb') as f:
    pickle.dump(lr_clf,f)

import json
columns = {
    'data_columns' : [col.lower() for col in x.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))

