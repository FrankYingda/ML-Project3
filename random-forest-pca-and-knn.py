import numpy as np
import pandas as pd
import seaborn as sb
sb.set_style("dark")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

def evaluate_classifier(clf, data, target, split_ratio):
    trainX, testX, trainY, testY = train_test_split(data, target, train_size=split_ratio, random_state=0)
    clf.fit(trainX, trainY)
    return clf.score(testX,testY)


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
target = train["label"]
train = train.drop("label", 1)

# plot first 64 digit number
plt.figure(figsize=(5,5))
for digit_num in range(0,64):
    plt.subplot(8,8,digit_num+1)
    grid_data = train.iloc[digit_num].as_matrix().reshape(28,28)  # reshape from 1d to 2d pixel array
    plt.imshow(grid_data, interpolation = "none", cmap = "bone_r")
    plt.xticks([])
    plt.yticks([])

plt.show()


# check performance of random forest classifier
n_estimators_array = np.array([1,5,10,50,100,200,500])
n_samples = 10
n_grid = len(n_estimators_array)
score_array_mu = np.zeros(n_grid)
score_array_sigma = np.zeros(n_grid)
j=0
for n_estimators in n_estimators_array:
    score_array = np.zeros(n_samples)
    for i in range(0, n_samples):
        clf = RandomForestClassifier(n_estimators = n_estimators, n_jobs=1, criterion="gini")
        score_array[i] = evaluate_classifier(clf, train.iloc[0:1000], target.iloc[0:1000], 0.8)
    score_array_mu[j], score_array_sigma[j] = np.mean(score_array), np.std(score_array)
    j=j+1

plt. figure(figsize=(7,3))
plt.errorbar(n_estimators_array, score_array_mu, yerr=score_array_sigma, fmt='k.-')
plt.xscale("log")
plt.xlabel("number of estimators",size = 20)
plt.ylabel("accuracy",size = 20)
plt.xlim(0.9,600)
plt.grid(which="both")
plt.show()


# find the most important features
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for f in range(0,10):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances

plt.figure(figsize=(7,3))
plt.plot(indices[:],importances[indices[:]],'k.')
plt.yscale("log")
plt.xlabel("feature",size=20)
plt.ylabel("importance",size=20)
plt.show()

# virsualize PCA
pca = PCA(n_components=2)
pca.fit(train)
transform = pca.transform(train)

plt.figure(figsize=(6,5))
plt.scatter(transform[:,0],transform[:,1], s=20, c = target, cmap = "nipy_spectral", edgecolor = "None")
plt.colorbar()
plt.clim(0,9)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


# number of PCA component
n_components_array=([1,2,3,4,5,10,20,50,100,200,500])
vr = np.zeros(len(n_components_array))
i=0;
for n_components in n_components_array:
    pca = PCA(n_components=n_components)
    pca.fit(train)
    vr[i] = sum(pca.explained_variance_ratio_)
    i=i+1

plt.figure(figsize=(8,4))
plt.plot(n_components_array,vr,'k.-')
plt.xscale("log")
plt.ylim(9e-2,1.1)
plt.yticks(np.linspace(0.2, 1.0, 9))
plt.xlim(0.9)
plt.grid(which="both")
plt.xlabel("number of PCA components",size=20)
plt.ylabel("variance ratio",size=20)
plt.show()


# KNN after PCA
clf = KNeighborsClassifier()
n_components_array=([1,2,3,4,5,10,20,50,100,200,500])
score_array = np.zeros(len(n_components_array))
i=0

for n_components in n_components_array:
    pca = PCA(n_components=n_components)
    pca.fit(train)
    transform = pca.transform(train.iloc[0:1000])
    score_array[i] = evaluate_classifier(clf, transform, target.iloc[0:1000], 0.8)
    i=i+1

plt.figure(figsize=(8,4))
plt.plot(n_components_array,score_array,'k.-')
plt.xscale("log")
plt.xlabel("number of PCA components",size=20)
plt.ylabel("accuracy", size=20)
plt.grid(which="both")
plt.show()


# compare with project1 which is just knn without PCA
X_train, X_test, X_train_Label, X_test_Label = train_test_split(train, target,
                                                 train_size=0.9, test_size=0.1,
                                                 random_state=1)

pca = PCA(n_components=50)
pca.fit(X_train)
transform_train = pca.transform(X_train)
transform_test = pca.transform(X_test)

KNC = KNeighborsClassifier()
KNC.fit(transform_train, X_train_Label)
results=KNC.predict(transform_test)
# print(results)
# print(X_test_Label)

print('Accuracy of K-NN classifier on training set: {:.2f}'.format(KNC.score(transform_train, X_train_Label)))
print('Accuracy of K-NN classifier on test set: {:.2f}'.format(KNC.score(results, X_test_Label)))



# save the file and upload to kaggle to test

# pca = PCA(n_components=50)
# pca.fit(train)
# transform_train = pca.transform(train)
# transform_test = pca.transform(test)
#
# clf = KNeighborsClassifier()
# clf.fit(transform_train, target)
# results=clf.predict(transform_test)
#
# # prepare submit file
#
# np.savetxt('pca_results.csv',
#            np.c_[range(1,len(test)+1),results],
#            delimiter=',',
#            header = 'ImageId,Label',
#            comments = '',
#            fmt='%d')
#


# clf = RandomForestClassifier(n_estimators = 100, n_jobs=1, criterion="gini")
# clf.fit(train, target)
# results=clf.predict(test)

# np.savetxt('rfc_results.csv',
#            np.c_[range(1,len(test)+1),results],
#            delimiter=',',
#            header = 'ImageId,Label',
#            comments = '',
#            fmt='%d')


















# 显示数字
# x_test = np.array(test)
#
# fig, ax = plt.subplots(10, 10, squeeze=True, figsize=(24,12))
# box = dict(facecolor='yellow', pad=5, alpha=1)
#
# for n in range(10):
#     for m in range(10):
#         ax[n][m].imshow(x_test[n*10+m].reshape(28,28), cmap='gray')
#         ax[n][m].set_title(results[[n*10+m]],y=0.9,bbox=box)
# plt.show()


