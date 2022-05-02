from pandas import read_csv, get_dummies, to_numeric
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt
from numpy import array

initial_data = read_csv('StudentsPerformanceMuricanUniversity.csv')
data = initial_data.replace('no data')
data['reading_score'] = to_numeric(data['reading_score'])
data = data.ffill().bfill()
data_dummies = get_dummies(data)

data_scores = data[['math_score', 'reading_score', 'writing_score']]

scaler = StandardScaler()
data = scaler.fit_transform(data_dummies)
data = normalize(data)
data = DataFrame(data)

data_scores = scaler.fit_transform(data_scores)
data_scores = normalize(data_scores)
data_scores = DataFrame(data_scores)

pca = PCA(n_components = 2)
data = pca.fit_transform(data)
data = DataFrame(data)

data_scores = pca.fit_transform(data_scores)
data_scores = DataFrame(data_scores)

epsilon = 0.12
min_samples = 15
metric = 'euclidean'            #‘cityblock’, ‘cosine’, ‘euclidean’, ‘l1’, ‘l2’, ‘manhattan’
data_DBscan = DBSCAN(epsilon, min_samples, metric)
data_DBscan.fit_predict(data)
labels = data_DBscan.labels_

clusters_number = len(set(labels)) - (1 if -1 in labels else 0)
noise_number = list(labels).count(-1)

print('Ilość klastrów: ', clusters_number, '\nIlość elementów szumu', noise_number)

silhouette = metrics.silhouette_score(data_dummies, labels)
davies_bouldin = metrics.davies_bouldin_score(data_dummies, labels)

print('Indeks silhouette: ', silhouette, '\nIndeks Daviesa-Bouldina: ', davies_bouldin)

x = data.to_numpy()[:, 0]
y = data.to_numpy()[:, 1]

plt.figure(figsize=(14, 7))
plt.scatter(x, y, c = labels, cmap = 'gist_rainbow', s = 50, edgecolor = 'none')
plt.title('DBSCAN', fontsize = 20)
plt.show()

data_numpy = data_dummies.to_numpy()
data_in_clusters = []
for cluster in range(clusters_number):
    cluster_data = []
    for data_index in range(len(data_numpy)):
        if labels[data_index] == cluster:
            cluster_data.append(data_numpy[data_index])
    data_in_clusters.append(cluster_data)

info_table = []
for cluster in range(clusters_number):
    data_in_cluster = array(data_in_clusters[cluster])
    info_data = []
    for i in range(3):
        score_avg = sum(data_in_cluster[:, i]) / len(data_in_cluster)
        info_data.append(score_avg)
    for i in range(3, 20):
        percentage = sum(data_in_cluster[:, i]) / len(data_in_cluster)
        info_data.append(percentage)
    info_table.append(info_data)

info_table = DataFrame(info_table)
info_table.columns = ['math_score', 'reading_score', 'writing_score', 'gender_female',
       'gender_male', 'race/ethnicity_group A', 'race/ethnicity_group B',
       'race/ethnicity_group C', 'race/ethnicity_group D',
       'race/ethnicity_group E',
       'parental_level_of_education_associates degree',
       'parental_level_of_education_bachelors degree',
       'parental_level_of_education_high school',
       'parental_level_of_education_masters degree',
       'parental_level_of_education_some college',
       'parental_level_of_education_some high school', 'lunch_free/reduced',
       'lunch_standard', 'test_preparation_course_completed',
       'test_preparation_course_none']
print(info_table.to_string())