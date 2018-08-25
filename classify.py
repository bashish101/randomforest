import os
import cv2
import numpy as np
import pandas as pd

from randomforest import Forest

def accuracy(y_true, y_pred):
	y_true = np.array(y_true, dtype = np.int32)
	y_pred = np.array(y_pred, dtype = np.int32)
	total = len(y_true)
	correct = sum(y_true == y_pred)
	return correct / float(total) * 100. 

def load_data(filepath, mode = None):
	print('Load started with mode {}'.format(mode))
	dataframe = pd.read_csv(filepath, header = None)
	features, labels = dataframe[dataframe.columns[:-1]].values.astype(np.float32), \
			   dataframe[dataframe.columns[-1]].values.astype(np.chararray)
	
	classes = list(set(labels))
	for idx in range(len(labels)):
		labels[idx] = classes.index(labels[idx])
	
	return (features, labels, classes)

def shuffle_data(features, labels):
	shuffled_idx = list(range(len(features)))
	np.random.shuffle(shuffled_idx)
	out_features = np.array([features[idx] for idx in shuffled_idx])
	out_labels = np.array([labels[idx] for idx in shuffled_idx])
	return (out_features, out_labels)

def main():
	np.random.seed(1)

	print('Loading data...')
	data = load_data('./sonar.all-data.csv')
	
	features, labels, classes = data[0], data[1], data[2]
	features, labels = shuffle_data(features, labels)

	train_size = int(0.75 * len(features))
	test_size = len(features) - train_size

	train_features, train_labels = features[:train_size], labels[:train_size]
	test_features, test_labels = features[train_size:], labels[train_size:]

	print ('Train size = {0} Test size = {1}'.format(train_size, test_size))

	print('Creating Random Forest Classifier...')
	classifier = Forest(num_trees = 10, mode = 'classification', classes = range(len(classes)))

	classifier.fit(train_features, train_labels)

	print('Generating test predictions...')
	prediction = classifier.predict(test_features)

	score = accuracy(prediction, test_labels)
	
	print ('Ground truth: ', test_labels)
	print ('Prediction: ', prediction)
	print ('Accuracy: {0:.2f}%'.format(score))

if __name__ == '__main__':
	main()
