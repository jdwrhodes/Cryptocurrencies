#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from pandas import DataFrame, Series
#%%
genre_df: DataFrame = pd.read_csv('./resources/genre_ratings.csv')
genre_df.head()

#%%
plt.scatter(
    x=genre_df['avg_romance_rating'],
    y=genre_df['avg_scifi_rating']
    #c=
)

#%%
kmeans_model: KMeans = KMeans(n_clusters=2, random_state=0)

#%%
kmeans_model.fit(genre_df)

#%%
y = kmeans_model.predict(genre_df)

#%%
y

#%%
plt.scatter(
    x=genre_df['avg_romance_rating'],
    y=genre_df['avg_scifi_rating'],
    c=y
)

#%%
for i in range(1,10):
    kmeans_model: KMeans = KMeans(n_clusters=i, random_state=0)
    kmeans_model.fit(genre_df)
    y = kmeans_model.predict(genre_df)
    plt.scatter(
        x=genre_df['avg_romance_rating'],
        y=genre_df['avg_scifi_rating'],
        c=y)
    plt.show()

#%%
inertia = []
k = range(1, len(genre_df)-1)

for i in k:
    kmeans_model: KMeans = KMeans(n_clusters=i, random_state=0)
    kmeans_model.fit(genre_df)
    inertia.append(kmeans_model.inertia_)

#%%
#elbow_data = {'k': k, 'inertia': inertia}
#elbow_df: DataFrame = DataFrame(elbow_data)
plt.scatter(
    x=k,
    y= inertia
    )

