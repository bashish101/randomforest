import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor

from randomforest import Forest

def impute_NaNs(dataframe):
	for key in dataframe:
		if key != 'Image':
			dataframe[key].fillna(dataframe[key].mean(), inplace = True)
	return dataframe

def hist_equalize(img, num_bins = 256):
	hist, bins = np.histogram(img.flatten(), num_bins, normed = True)
	cdf = hist.cumsum()
	cdf = (cdf / cdf[-1]) * 255.
	
	out_img = np.interp(img.flatten(), bins[:-1], cdf)
	out_img = out_img.reshape(img.shape)
	return out_img

def mirror_data(img_data, lab_data, img_h = 96, img_w = 96):
	out_img_data = np.copy(img_data)
	out_lab_data = np.copy(lab_data)
	out_img_data = out_img_data.reshape(-1, img_h, img_w)
	
	if np.random.rand() < 1:
		# Vertical flip
		out_img_data = out_img_data[:, ::-1]
		out_lab_data[:, 1::2] = img_h - out_lab_data[:, 1::2]   
	else:
		# Horizontal flip
		out_img_data = out_img_data[:, :, ::-1]
		out_lab_data[:, 0::2] = img_w - out_lab_data[:, 0::2]
  
	out_img_data = out_img_data.reshape(-1, img_h * img_w)
	return (out_img_data, out_lab_data)

def load_data(filepath, mode = 'train', preprocess_flag = True, normalize_flag = False):
	print('Load started with mode {}'.format(mode))
	dataframe = pd.read_csv(filepath, header = 0)
	
	if mode == 'train':
		print('Handling NaNs...')
		if preprocess_flag:
			dataframe = impute_NaNs(dataframe)
		else:
			dataframe = dataframe.dropna()

	img_data = dataframe['Image'].apply(lambda im: np.fromstring(im, sep = ' '))
	img_data = np.vstack(img_data.values).astype(np.float32)
	
	if preprocess_flag:
		print('Equalizing histograms...')
		for idx in range(len(img_data)):
			img_data[idx] = hist_equalize(img_data[idx])
		if normalize_flag:		
			print('Normalizing data...')
			img_data -= np.mean(img_data, axis = 0)
			img_data /= np.std(img_data, axis = 0)

	if mode == 'train':
		lab_data = dataframe.drop(['Image'], axis = 1)
		lab_data = lab_data.values.astype(np.float32)

		if preprocess_flag:
			print('Performing data augmentation...')
			img_data_aug, lab_data_aug = mirror_data(img_data, lab_data)
			img_data = np.vstack((img_data, img_data_aug))
			lab_data = np.vstack((lab_data, lab_data_aug))
	else:
		lab_data = None

	print('Load completed with mode {}'.format(mode))
	return (img_data, lab_data)

def display(img_data, lab_data, img_h = 96, img_w = 96):
	num_points = int(len(lab_data[0]) / 2)
	img_data = img_data.reshape(-1, img_h, img_w)
	color = (0, 0, 255)
	for img, lab in zip(img_data, lab_data):
		img = img.astype(np.uint8)
		test = Image.fromarray(img)
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)	
		for idx in range(num_points):
			x, y = int(lab[idx * 2]), int(lab[idx * 2 + 1])
			cv2.putText(img, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
		img = img[:, :, ::-1]
		img = Image.fromarray(img)
		img.show()		
		input('Enter any key...')
		

def main():
	np.random.seed(1)

	print('Loading data...')
	train_data = load_data('./kaggle_data/training.csv', mode = 'train', preprocess_flag = False)
	test_data = load_data('./kaggle_data/test.csv', mode = 'test', preprocess_flag = False)
	X_train, Y_train = train_data[0], train_data[1]
	X_test, _ = test_data[0], test_data[1]

	inp_data = np.copy(X_test)

	print('Performing PCA...')
	pca = PCA(n_components = 23, svd_solver='randomized')
	pca.fit(X_train)
	
	X_train = pca.fit_transform(X_train)
	X_test = pca.fit_transform(X_test)

	print('Creating Random Forest Regressor...')

	# Uncomment to try sklearn's decision tree regressor
	# regressor = DecisionTreeRegressor(max_depth = 10)
	# regressor.fit(X_train, Y_train)

	regressor = Forest(num_trees = 1, mode = 'regression')
	regressor.fit(X_train, Y_train)

	print('Generating test predictions...')
	Y_test = regressor.predict(X_test)

	display(inp_data[-10:], Y_test[-10:])

if __name__ == '__main__':
	main()
