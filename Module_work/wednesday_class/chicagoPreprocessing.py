#%%
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from pandas import DataFrame, Series

#%%
marathon_df: DataFrame = pd.read_csv('./chicago_marathon_2018.csv')
marathon_df.head()

#%%
marathon_df['country'] = marathon_df['name'].str.extract('\((.{3})\)')

#%%
marathon_df.head()

#%%
marathon_df['country'].value_counts()[0:25]

#%%
marathon_df['country'] = marathon_df['country'].apply(lambda x: x if x in ('USA', 'MEX', 'GBR', 'CHN', 'CAN', 'BRA') else 'Other')

#%%
marathon_df.head()

#%%
marathon_df['country'].value_counts()

#%%
marathon_df.dtypes

#%%
marathon_df[['half', 'finish']] = marathon_df[['half', 'finish']].apply(pd.to_timedelta).apply(lambda x: x.dt.total_seconds())

#%%
marathon_df.head()

#%%
marathon_df['place_overall'].value_counts()
#%%
X = marathon_df[['half', 'finish', 'division', 'country']].copy()

#%%
X

#%%
X['division'] = LabelEncoder().fit_transform(X['division'])
X['country'] = LabelEncoder().fit_transform(X['country'])

#%%
X.dtypes

#%%
X.dropna()

#%%
X_scaled = MinMaxScaler().fit_transform(X)

#%%
X_scaled

#%%
X_scaled = pd.DataFrame(X_scaled)

#%%
X_scaled

#%%
X_scaled.values