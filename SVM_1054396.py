from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
# Importing the dataset
wq_ds = pd.read_csv('winequality-red.csv', sep=',')
y = wq_ds.quality
x = wq_ds.drop('quality', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Normalization
sc = StandardScaler()
x_train1 = sc.fit_transform(x_train)
x_test1 = sc.fit_transform(x_test)

# Implementing the Support Vector Machine model
svc_model_def = SVC()
svc_model_def.fit(x_train1, y_train)
svc_model_def_predict = svc_model_def.predict(x_test1)
print("Η απόδοση του μοντέλου μας για τις default παραμέτρους:")
print(classification_report(y_test, svc_model_def_predict, zero_division=0))
print(confusion_matrix(y_test, svc_model_def_predict))

# param_grid = {'C': [0.1, 1, 10, 100],
#                   'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001],
#                   'kernel': ['rbf', 'sigmoid', 'linear']}
# grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3, return_train_score=True)
# grid.fit(x_train, y_train)
#
# print(" C gamma and kernel καλύτεροι παράμετρες: ", grid.best_params_)
# print("Σκορ του ταξινομητή για τις καλύτερες παραμέτρους: ", grid.best_score_)
# print("Ο καλύτερος ταξινομητής: ", grid.best_estimator_)
# best_svm_clf = grid.best_estimator_

# SVM with the best parameters
svc_model = SVC(C=10, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
                       decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf',
                       max_iter=-1, probability=False, random_state=None, shrinking=True,
                       tol=0.001, verbose=False)
svc_model.fit(x_train1, y_train)
svc_model_predict = svc_model.predict(x_test1)

print("Η απόδοση του μοντέλου μας για τις καλύτερες παραμέτρους:")
print(classification_report(y_test, svc_model_predict, zero_division=0))
print(confusion_matrix(y_test, svc_model_predict))

# Second query. Removing 33% of ph rows from the training dataset and handling these missing values with 4 different methods
# Removing 33% of ph values
wq_ds1 = x_train.pH
wq_ds2 = wq_ds1.sample(frac=.33, random_state=138)
wq_ds1 = wq_ds1.drop(wq_ds2.index)
wq_ds3 = x_train.drop('pH', axis=1)
wq_ds4 = wq_ds3.join(wq_ds1)

# SVC with no ph column
x_train2 = wq_ds4.drop('pH', axis=1)
x_test2 = x_test.drop('pH', axis=1)
x_train2 = sc.fit_transform(x_train2)
x_test2 = sc.fit_transform(x_test2)
svc_model.fit(x_train2, y_train)
svc_model_predict_noph = svc_model.predict(x_test2)
print("Η απόδοση του μοντέλου μας για τις καλύτερες παραμέτρους του SVM και χωρίς πλέον την στήλη pH:")
print(classification_report(y_test, svc_model_predict_noph, zero_division=0))
print(confusion_matrix(y_test, svc_model_predict_noph))

# Filling NaN values with the mean of the rest.
x_train3 = wq_ds4.fillna(x_train.pH.mean())
x_train3 = sc.fit_transform(x_train3)
svc_model.fit(x_train3, y_train)
svc_model_predict_avg = svc_model.predict(x_test1)
print("Η απόδοση του μοντέλου μας για τις καλύτερες παραμέτρους του SVM και όπου NaΝ στο pΗ πλέον υπάρχει ο μέσος όρος των υπόλοιπων τιμών")
print(classification_report(y_test, svc_model_predict_avg, zero_division=0))
print(confusion_matrix(y_test, svc_model_predict_avg))


# Filling NaN values using Logistic Regression.
x_train_nan = wq_ds4.isnull()
x_train_rows_nan = x_train_nan.any(axis=1)
x_train_rows_with_nan = wq_ds4[x_train_rows_nan]
wq_ds4_no_nan = wq_ds4.dropna()

# Logistic Regression Model
y1 = wq_ds4_no_nan.pH
lab_enc = preprocessing.LabelEncoder()
y1 = lab_enc.fit_transform(y1)
x1 = wq_ds4_no_nan.drop('pH', axis=1)
# Splitting to train and test
x_train_lr, x_test_lr, y_train_lr, y_test_lr = train_test_split(x1, y1, test_size=0.25, random_state=0)
lr = LogisticRegression(random_state=0, solver="liblinear")
lr.fit(x_train_lr, y_train_lr)
lr_predict = lr.predict(x_train_rows_with_nan.drop('pH', axis=1))
predictionPH = lab_enc.inverse_transform(lr_predict)
predictionPH = np.reshape(predictionPH, (predictionPH.shape[0], 1))

# nparray to DataFrame
predictionPH = pd.DataFrame(predictionPH, columns=['pH'])
#x_train_rows_with_nan.append(predictionPH)

x_train_rows_with_nan['pH'] = predictionPH['pH'].values
frames = [wq_ds4_no_nan, x_train_rows_with_nan]

result = pd.concat(frames)

result.sort_index(inplace=True)
y_train.sort_index(inplace=True)
result = sc.fit_transform(result)
svc_model.fit(result, y_train)
svc_model_predict = svc_model.predict(x_test1)

print("Η απόδοση του logistic regression μας για τις καλύτερες παραμέτρους:")
print(classification_report(y_test, svc_model_predict, zero_division=0))
print(confusion_matrix(y_test, svc_model_predict))


# Filling NaN values with kmeans
# Finding the best #of clusters for our dataset
# distortion = []
# K = range(1,10)
# for k in K:
#     kmean_Model = KMeans(n_clusters=k).fit(x_train)
#     kmean_Model.fit(x_train)
#     distortion.append(sum(np.min(cdist(x_train, kmean_Model.cluster_centers_, 'euclidean'), axis=1)) / x_train.shape[0])
# # Plot the elbow
# plt.plot(K, distortion, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Παραμόρφωση')
# plt.title('Η Elbow Method που δείχνει το καλύτερο k')
# plt.show()

# initializing kmeans
kmeans = KMeans(n_clusters=3, init="k-means++", max_iter=300)
kmeans.fit(wq_ds4_no_nan)
# labels of kmeans
labels = kmeans.labels_
# centroids of the clusters
centroids = kmeans.cluster_centers_
# True for NaN, False for actual value
nan_or_not = ~np.isfinite(wq_ds4)
# Mean of every column
mean = np.nanmean(wq_ds4, 0, keepdims=True)
# If False, choose mean. If True, choose wq_ds4 value
new_dataset = np.where(nan_or_not, mean, wq_ds4)
max_iter = 10
for i in range(max_iter):
    if i > 0:
        method = KMeans(3, init=centroids)
    else:

        method = KMeans(3, n_jobs=-1)

    # Clustering for the new data
    labels1 = method.fit_predict(new_dataset)
    # New centroids
    centroids = method.cluster_centers_
    # fill in the missing values based on their cluster centroids

    new_dataset[nan_or_not] = centroids[labels1][nan_or_not]

    # when the labels have stopped changing then we have converged
    if i > 0 and np.all(labels1 == labels):
        break

prev_labels = labels1
prev_centroids = method.cluster_centers_
new_dataset = pd.DataFrame(new_dataset, columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'sulphates', 'alcohol', 'pH'])
x_train_new = new_dataset
y_new = y_train
x_train, x_test, y_train_new, y_test_new = train_test_split(x_train_new, y_new, test_size=0.25, random_state=0)
x_train1 = sc.fit_transform(x_train)
x_test1 = sc.fit_transform(x_test)
svc_model.fit(x_train1, y_train_new)
svc_model_predict_Kmean = svc_model.predict(x_test1)
print("Η απόδοση του μοντέλου μας για τις καλύτερες παραμέτρους του SVM και όπου NaΝ στο pΗ πλέον υπάρχει ο αριθμητικός μέσος όρος του cluster στο οποίο ανήκει κάθε δείγμα")
print(classification_report(y_test_new, svc_model_predict_Kmean, zero_division=0))
print(confusion_matrix(y_test_new, svc_model_predict_Kmean))