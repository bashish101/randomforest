import numpy as np

class Tree(object):
	def __init__(self,
		     params = {
				     'mode' : 'regression',
				     'classes' : None,
				     'comparison' : 'axisaligned',
				     'selection_count' : None,
				     'minimum_count' : 1,
				     'maximum_depth' : 10,
				     'threshold_count' : 100
				}):
		
		self.mode = params['mode']
		self.comparison = params['comparison']
		self.selection_count = params['selection_count']
		self.minimum_count = params['minimum_count']
		self.maximum_depth = params['maximum_depth']
		self.threshold_count = params['threshold_count']
		self.classes = params['classes']

		if self.mode == 'regression':
			self.eval_fn = self.regress
			self.score_fn = self.mse
		elif self.mode == 'classification':
			self.eval_fn = self.classify
			self.score_fn = self.gini

		if self.comparison == 'linear':
			self.compare_fn = self.linear
		elif self.comparison == 'conic':
			self.compare_fn = self.conic
		elif self.comparison == 'parabola':
			self.compare_fn = self.parabola
		else:
			self.compare_fn = self.axis_aligned

		self.root = self.get_node()

	def get_node(self):
		return dict({'leaf' : None, 
			     'left' : None, 
			     'right' : None, 
			     'threshold' : None})

	def mse(self, labels):
		size = len(labels)
		if size == 0:
			return 0.
		mean = np.mean(labels, axis = 0)
		return np.mean((labels - mean) ** 2)

	def gini(self, labels):
		size = len(labels)
		if size == 0:
			return 0.

		if self.classes is None:
			classes = set(labels)
		else:
			classes = self.classes

		score = 0.
		for class_val in classes:
			pb = list(labels).count(class_val) / float(size)
			score += pb ** 2
		return (1. - score)

	def regress(self, labels):
		return np.mean(labels, axis = 0)

	def classify(self, labels):
		return max(set(labels), key = list(labels).count)

	def axis_aligned(self, feat_row, threshold):
		return (feat_row[threshold['index']] < threshold['value'])

	def linear(self, feat_row, threshold):
		pass

	def conic(self, feat_row, threshold):
		pass

	def parabola(self, feat_row, threshold):
		pass

	def generate_thresholds(self, features, count):
		min_val = min(features)
		max_val = max(features)
		return np.linspace(min_val, max_val, count)

	def split(self, features, labels, threshold):
		left = []
		right = []
		for feat_row, label in zip(features, labels):
			if self.compare_fn(feat_row, threshold):
				left.append(label)
			else:
				right.append(label)
		return left, right

	def get_split_point(self, features, labels):
		selection = []
		if self.selection_count is not None:
			while(len(selection) < self.selection_count):
				feat_idx = np.random.randint(len(features[0]))
				if feat_idx not in selection:
					selection.append(feat_idx)
			
		else:
			selection.extend(range(len(features[0])))

		best_error = np.inf
		best_thresh = None
		for feat_idx in selection:
			thresholds = self.generate_thresholds(features[:, feat_idx], self.threshold_count)
			for idx, value in enumerate(thresholds):
				threshold = {'value' : value, 'index' : feat_idx}
				left, right = self.split(features, labels, threshold)
				error = (len(left) / float(len(features)) * self.score_fn(left)) + \
					(len(right) / float(len(features)) * self.score_fn(right))
				if error < best_error:
					best_error = error
					best_thresh = threshold

		return best_thresh
		

	def make_leaf(self, node, labels):
		node['leaf'] = self.eval_fn(labels)
	
	def create_tree(self, node, features, labels, depth):
		features = np.array(features)
		labels = np.array(labels)
		error = self.score_fn(labels)

		if (depth == self.maximum_depth or 
		    len(features) <= self.minimum_count or
		    error == 0.):
			self.make_leaf(node, labels)
			return

		threshold = self.get_split_point(features, labels)
		if threshold is None:
			self.make_leaf(node, labels)
			return

		node['threshold'] = threshold

		left_features = []
		left_labels = []
		right_features = []
		right_labels = []
		for feat_row, label in zip(features, labels):
			if self.compare_fn(feat_row, node['threshold']):
				left_features.append(feat_row)
				left_labels.append(label)
			else:
				right_features.append(feat_row)
				right_labels.append(label)

		node['left'] = self.get_node()
		node['right'] = self.get_node()

		self.create_tree(node['left'], left_features, left_labels, depth + 1)
		self.create_tree(node['right'], right_features, right_labels, depth + 1)

	def fit(self, features, labels):
		self.create_tree(self.root, features, labels, depth = 0)	

	def traverse_tree(self, node, feat_row):
		if node['leaf'] is not None:
			return node['leaf']
		else:
			if self.compare_fn(feat_row, node['threshold']):
				return self.traverse_tree(node['left'], feat_row)
			else:
				return self.traverse_tree(node['right'], feat_row)

	def predict(self, feat_row):
		return self.traverse_tree(self.root, feat_row)				
				

