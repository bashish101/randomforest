import os
import math
import numpy as np
from tree import Tree

class Forest(object):
	def __init__(self,
		     num_trees = 10,
		     subsample_flag = True,
		     mode = 'regression',
		     classes = None,
		     comparison = 'axisaligned',
		     selection_count = None,
		     minimum_count = 1,
		     maximum_depth = 10,
		     threshold_count = 100):

		if mode == 'regression':
			self.eval_fn = self.regress
		else:
			self.eval_fn = self.classify

		self.num_trees = num_trees
		self.params = {
				     'mode' : mode,
				     'classes' : classes,
				     'comparison' : comparison,
				     'selection_count' : selection_count,
				     'minimum_count' : minimum_count,	
				     'maximum_depth' : maximum_depth,				
				     'threshold_count' : threshold_count
				}
		self.subsample_flag = subsample_flag
		self.trees = []

	def __len__(self):
		return self.num_trees

	def save(self, folder):
		pass

	def load(self, folder):
		pass

	def regress(self, labels):
		return np.mean(labels, axis = 0)

	def classify(self, labels):
		return max(set(labels), key = labels.count)

	def subsample(self, features, labels, ratio = 0.8):
		sampled_features = []
		sampled_labels = []
		size = round(len(features) * ratio)
		while len(sampled_features) < size:
			idx = np.random.randint(len(features))
			sampled_features.append(features[idx])
			sampled_labels.append(labels[idx])

		return (sampled_features, sampled_labels)

	def fit(self, features, labels):
		if self.params['selection_count'] is None:
			self.params['selection_count'] = int(math.sqrt(len(features[0])))

		for idx in range(self.num_trees):
			self.trees.append(Tree(self.params))

			if self.subsample_flag == True:
				sampled_features, sampled_labels = self.subsample(features, labels)
				self.trees[idx].fit(sampled_features, sampled_labels)
			else:
				self.trees[idx].fit(features, labels)

	def predict(self, data):
		labels = []
		for feat_row in data:
			prediction = []
			for idx in range(self.num_trees):
				prediction.append(self.trees[idx].predict(feat_row))
			labels.append(self.eval_fn(prediction))
		return labels
