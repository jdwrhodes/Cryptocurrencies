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
X_scaled: DataFrame = pd.DataFrame(X_scaled)
#%%
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
# %%
X_scaled = X_scaled.dropna()

#%%

inertia = {}

K = range(1,10)

for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X_scaled)
    inertia[k] = kmeanModel.inertia_
#%%
plt.plot(list(inertia.keys()), list(inertia.values()))
# %%
kmeansModel: KMeans = KMeans(n_clusters=3, random_state=0).fit(X_scaled)
# %%
y_pred: Series = pd.Series(kmeansModel.predict(X_scaled))
# %%
y_pred
#%%
X
X = X.dropna()
#%%
plt.scatter(
    x=X['half'],
    y=X['finish'],
    c=y_pred)
# %%
plt.show()

#%%
from sklearn.decomposition import PCA

#%%
pca: PCA = PCA(n_components=2)

#%%
pca.fit(X_scaled)

#%%
pca.explained_variance_ratio_

#%%
X_pca = pca.transform(X_scaled)

#%%
X_pca: DataFrame = pd.DataFrame(X_pca)

#%%
X_pca

#%%

inertia = {}

K = range(1,10)

for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X_pca)
    inertia[k] = kmeanModel.inertia_

plt.plot(list(inertia.keys()), list(inertia.values()))
# %%
kmeansModel: KMeans = KMeans(n_clusters=7, random_state=0).fit(X_pca)
# %%
y_pred: Series = pd.Series(kmeansModel.predict(X_pca))
# %%
X = X.dropna()

#%%
plt.scatter(
    x=X_pca[0],
    y=X_pca[1],
    c=y_pred)