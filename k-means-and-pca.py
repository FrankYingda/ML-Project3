import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, \
    adjusted_mutual_info_score, silhouette_score
import warnings
warnings.filterwarnings('ignore')


#np.random.seed(123)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth',20)
pd.set_option('display.width', 150)

Data = pd.read_csv('train.csv')
# print('Shape of the data set: ' + str(Data.shape))
#
# print(Data.sample(1))

Labels = Data['activity']
Data_new = Data.drop(['rn', 'activity'], axis = 1)
Labels_keys = Labels.unique().tolist()
Labels = np.array(Labels)

# print('Activity labels: ' + str(Labels_keys))
# Temp = pd.DataFrame(Data_new.isnull().sum())
# Temp.columns = ['Sum']
# print('Amount of rows with missing values: ' + str(len(Temp.index[Temp['Sum'] > 0])) )


scaler = StandardScaler()
Data_new1 = scaler.fit_transform(Data_new)
#print(Data_new)


# "elbow" of the line
ks = range(1, 10)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(Data_new)
    inertias.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.style.use('bmh')
plt.plot(ks, inertias, '-o')
plt.xlabel('Number of clusters, k')
plt.ylabel('Inertia')
plt.xticks(ks)
plt.show()


# calinski_harabaz_score & silhouette_score
# for j in  range(2, 7):
#     model_kmeans=KMeans(n_clusters=j, random_state=123, n_init=10)
#     model_kmeans.fit(Data_new)
#     labels = model_kmeans.labels_
#     chs = metrics.calinski_harabaz_score(Data_new, labels)
#     print("when k is ", j, ", calinski_harabaz_score is ", chs)
#     sis = metrics.silhouette_score(Data_new, labels)
#     print("when k is ", j, ", silhouette_score is ", sis, "\n")

#
def k_means(n_clust, data_frame, true_labels):
    k_means = KMeans(n_clusters=n_clust, random_state=123, n_init=30)
    k_means.fit(data_frame)
    c_labels = k_means.labels_
    df = pd.DataFrame({'clust_label': c_labels, 'orig_label': true_labels.tolist()})
    ct = pd.crosstab(df['clust_label'], df['orig_label'])
    y_clust = k_means.predict(data_frame)
    display(ct)
    print()
    print("cluster number is :", n_clust)
    print('% 9s' % 'inertia  homo    compl   v-meas   ARI     AMI     silhouette    calinski')
    print('%i   %.3f   %.3f   %.3f   %.3f   %.3f   %.3f         %.3f'
          % (k_means.inertia_,
             homogeneity_score(true_labels, y_clust),
             completeness_score(true_labels, y_clust),
             v_measure_score(true_labels, y_clust),
             adjusted_rand_score(true_labels, y_clust),
             adjusted_mutual_info_score(true_labels, y_clust),
             silhouette_score(data_frame, y_clust, metric='euclidean'),
             metrics.calinski_harabaz_score(data_frame, y_clust)
          ))

# k_means(n_clust=6, data_frame=Data_new, true_labels=Labels)



Labels_binary = Labels.copy()
for i in range(len(Labels_binary)):
    if (Labels_binary[i] == 'STANDING' or Labels_binary[i] == 'SITTING' or Labels_binary[i] == 'LAYING'):
        Labels_binary[i] = 0
    else:
        Labels_binary[i] = 1

Labels_binary = np.array(Labels_binary.astype(int))
#
# k_means(n_clust=2, data_frame=Data_new, true_labels=Labels_binary)


# pca = PCA(random_state=123)
# pca.fit(Data_new)
# features = range(pca.n_components_)
#
# plt.figure(figsize=(8,4))
# plt.bar(features[:15], pca.explained_variance_[:15], color='lightskyblue')
# plt.xlabel('PCA feature')
# plt.ylabel('Variance')
# plt.xticks(features[:15])
# plt.show()
#

def pca_transform(n_comp):
    pca = PCA(n_components=n_comp, random_state=123)
    global Data_reduced
    Data_reduced = pca.fit_transform(Data_new1)
    print('Shape of the new Data df: ' + str(Data_reduced.shape))

pca_transform(n_comp=1)

k_means(n_clust=2, data_frame=Data_reduced, true_labels=Labels_binary)
