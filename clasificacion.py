from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib

fixed_size = tuple((500,500))
num_trees = 100
bins = 8
test_size = 0.10
seed = 9

# feature-desc 1: Invariante Hu
def fd_hu_moments(image_var):
  image_var = cv2.cvtColor(image_var, cv2.COLOR_BGR2GRAY)
  feature = cv2.HuMoments(cv2.moments(image_var)).flatten()
  return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image_var):
# convertimos la imagen a campo gris
  gray = cv2.cvtColor(image_var, cv2.COLOR_BGR2GRAY)
# obtenemos un vector
  haralick = mahotas.features.haralick(gray).mean(axis=0)
  return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image_var, mask=None):
# convertimos la imagen hacia HSV color-space
  image_var = cv2.cvtColor(image_var, cv2.COLOR_BGR2HSV)
# obtenemos el histograma de color de pixeles
  hist = cv2.calcHist([image_var], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
# normalizamos el histograma
  cv2.normalize(hist, hist)
  return hist.flatten()

os.chdir('/content/train')
os.getcwd()
os.listdir()
train_labels = os.listdir()
train_labels.sort()
train_labels.pop(0) 
print(train_labels)
os.getcwd()

global_features = []
labels = []

i,j,k = 0,0,0

images_per_class = 78

from glob import glob
train_path = '/content/train'

for training_name in train_labels:
    dir = os.path.join(train_path, training_name)
    print(dir)
    jat=[]
    jat.clear()
    os.chdir(dir)
    jat=glob('*.jpg')
    print(jat)
    print("la cantidad de elementos en la carpeta " + training_name +" es : " + str(len(jat)))
    current_label = training_name
    k = 1
    for x in range(len(jat)) :
        nombre = str(jat[x])
        file = dir + "//" + nombre 
        image = cv2.imread(file)
        image = cv2.resize(image, fixed_size)

        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)

        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        labels.append(current_label)
        global_features.append(global_feature)

        i += 1
        k += 1
    print ("[STATUS] processed folder: {}".format(current_label))
    j += 1

print ("[STATUS] completed Global Feature Extraction...")
print ("[STATUS] feature vector size {}".format(np.array(global_features).shape))
print ("[STATUS] training Labels {}".format(np.array(labels).shape))

targetNames = np.unique(labels)
le = LabelEncoder()
target = le.fit_transform(labels)
print ("[STATUS] training labels encoded...")

scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print ("[STATUS] feature vector normalized...")

print ("[STATUS] target labels: {}".format(target))
print ("[STATUS] target labels shape: {}".format(target.shape))


output_path ='/content/output'
os.chdir(output_path)
h5f_data = h5py.File('data.h5', 'w')
h5f_data.create_dataset('dataset', data=np.array(rescaled_features))

h5f_label = h5py.File('labels.h5', 'w')
h5f_label.create_dataset('dataset', data=np.array(target))

h5f_data.close()
h5f_label.close()

print ("[STATUS] end of training..")

models = []
models.append(('LR', LogisticRegression(random_state=9)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=9)))
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=9)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(random_state=9)))

results = []
names = []
scoring = "accuracy"

h5f_data = h5py.File('data.h5', 'r')
h5f_label = h5py.File('labels.h5', 'r')

global_features_string = h5f_data['dataset']
global_labels_string = h5f_label['dataset']

global_features = np.array(global_features_string)
global_labels = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

print ("[STATUS] features shape: {}".format(global_features.shape))
print ("[STATUS] labels shape: {}".format(global_labels.shape))
print ("[STATUS] training started...")

(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=test_size,
                                                                                          random_state=seed)

print ("[STATUS] splitted train and test data...")
print ("Train data  : {}".format(trainDataGlobal.shape))
print ("Test data   : {}".format(testDataGlobal.shape))
print ("Train labels: {}".format(trainLabelsGlobal.shape))
print ("Test labels : {}".format(testLabelsGlobal.shape))


import warnings
warnings.filterwarnings('ignore')

for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

fig = pyplot.figure()
fig.suptitle('Comparacion de algoritmos de clasificación')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

import matplotlib.pyplot as plt

clf  = RandomForestClassifier(n_estimators=100, random_state=9)
#clf  = GaussianNB(priors=None)

clf.fit(trainDataGlobal, trainLabelsGlobal)


test_path ='/content/test'

for file in glob(test_path + "/*.jpg"):
    image = cv2.imread(file)
    image = cv2.resize(image, fixed_size)

    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image)
    fv_histogram  = fd_histogram(image)

    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

    prediction = clf.predict(global_feature.reshape(1,-1))[0]
    
    cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,255), 3)
    print(file)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()