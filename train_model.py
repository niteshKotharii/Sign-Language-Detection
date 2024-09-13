import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./data_merged.pickle', 'rb'))

data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

clf1 = RandomForestClassifier().fit(x_train, y_train)
clf2 = KNeighborsClassifier(n_neighbors=7).fit(x_train, y_train)
clf3 = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300).fit(x_train, y_train)

y_pred1 = clf1.predict(x_test)
y_pred2 = clf2.predict(x_test)
y_pred3 = clf3.predict(x_test)

score1 = accuracy_score(y_pred1, y_test)
score2 = accuracy_score(y_pred2, y_test)
score3 = accuracy_score(y_pred3, y_test)

print('{}% accuracy for randomForest !'.format(score1 * 100))
print('{}% accuracy for KNN !'.format(score2 * 100))
print('{}% accuracy for MLC !'.format(score3 * 100))

f = open('model.p', 'wb')
pickle.dump({'model': clf3}, f)
f.close()
