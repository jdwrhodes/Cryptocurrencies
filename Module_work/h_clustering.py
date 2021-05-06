#%%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import hvplot.pandas
# %%
# Load the data
file = './resources/new_iris_data.csv'
df_iris = pd.read_csv(file)
df_iris.head()

#%%
import plotly.figure_factory as ff 

#%%
iris_scaled = StandardScaler().fit_transform(df_iris)
print(iris_scaled[0:5])

#%%
# Initialize the PCA model
pca = PCA(n_components=2)

#%%
# Use PCA to reduce the dimensions from 4 to 2
iris_pca = pca.fit_transform(iris_scaled)

#%%
# Transform the PCA into a DF
df_iris_pca = pd.DataFrame(data=iris_pca, columns=['principal component 1', 'principal component 2'])
df_iris_pca.head()
#%%
# Create a dendrogram
fig = ff.create_dendrogram(df_iris_pca, color_threshold=0)
fig.update_layout(width=800, height=500)
fig.show()

#%%
agg = AgglomerativeClustering(n_clusters=3)
model = agg.fit(df_iris_pca)

#%%
# Add a new class column to df_iris
df_iris_pca['class'] = model.labels_
df_iris_pca.head()

#%%
df_iris_pca.hvplot.scatter(
    x='principal component 1',
    y='principal component 2',
    hover_cols=['class'],
    by='class'
)