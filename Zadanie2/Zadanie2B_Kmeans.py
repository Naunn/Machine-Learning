from pandas import read_csv, get_dummies, to_numeric
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt
from numpy import array

def kmeans_2(data, n_clusters, metric, iterations):
    centroids = [data.sample().to_numpy()[0] for _ in range(n_clusters)]
    for iteration in range(iterations):
        labels = []
        for index, row in data.iterrows():
            if metric == 'euclidean':
                distances = [(row[0] - centroids[i][0]) ** 2 + (row[1] - centroids[i][1]) ** 2 for i in range(len(centroids))]
            if metric == 'manhattan':
                distances = [abs(row[0] - centroids[i][0]) + abs(row[1] - centroids[i][1]) for i in range(len(centroids))]
            if metric == 'czebyszew':
                distances = [max(abs(row[0] - centroids[i][0]), abs(row[1] - centroids[i][1])) for i in range(len(centroids))]
            minimum = min(distances)
            labels.append(distances.index(minimum))
        for cluster in range(n_clusters):
            cluster_data = []
            for data_index in range(data.shape[0]):
                if labels[data_index] == cluster:
                    cluster_data.append(data.to_numpy()[data_index])
            cluster_data = array(cluster_data)
            new_centroid = [sum(cluster_data[:, 0]) / len(cluster_data), sum(cluster_data[:, 1]) / len(cluster_data)]
            centroids[cluster] = new_centroid
    return labels

def kmeans_3(data, n_clusters, metric, iterations):
    centroids = [data.sample().to_numpy()[0] for _ in range(n_clusters)]
    for iteration in range(iterations):
        labels = []
        for index, row in data.iterrows():
            if metric == 'euclidean':
                distances = [(row[0] - centroids[i][0]) ** 2 + (row[1] - centroids[i][1]) ** 2 + (row[2] - centroids[i][2]) ** 2 for i in range(len(centroids))]
            if metric == 'manhattan':
                distances = [abs(row[0] - centroids[i][0]) + abs(row[1] - centroids[i][1]) + abs(row[2] - centroids[i][2]) for i in range(len(centroids))]
            if metric == 'czebyszew':
                distances = [max(abs(row[0] - centroids[i][0]), abs(row[1] - centroids[i][1]), abs(row[2] - centroids[i][2])) for i in range(len(centroids))]
            minimum = min(distances)
            labels.append(distances.index(minimum))
        for cluster in range(n_clusters):
            cluster_data = []
            for data_index in range(data.shape[0]):
                if labels[data_index] == cluster:
                    cluster_data.append(data.to_numpy()[data_index])
            cluster_data = array(cluster_data)
            new_centroid = [sum(cluster_data[:, 0]) / len(cluster_data), sum(cluster_data[:, 1]) / len(cluster_data), sum(cluster_data[:, 2]) / len(cluster_data)]
            centroids[cluster] = new_centroid
    return labels

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

pca = PCA(n_components = 2)
data = pca.fit_transform(data)
data = DataFrame(data)

clusters_number = 5
metric = 'czebyszew'
iterations = 1000
labels = kmeans_2(data, clusters_number, metric, iterations)
print(labels)

silhouette = metrics.silhouette_score(data_dummies, labels)
davies_bouldin = metrics.davies_bouldin_score(data_dummies, labels)

print('Indeks silhouette: ', silhouette, '\nIndeks Daviesa-Bouldina: ', davies_bouldin)

x = data.to_numpy()[:, 0]
y = data.to_numpy()[:, 1]

plt.figure(figsize=(14, 7))
plt.scatter(x, y, c = labels, cmap = 'gist_rainbow', s = 50, edgecolor = 'none')
plt.title('K-means', fontsize = 20)
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