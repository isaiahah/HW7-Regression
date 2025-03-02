"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
from regression import LogisticRegressor
import numpy as np
# (you will probably need to import more things here)

def test_prediction():
	# Test the prediction on a small dummy example
	log_model = LogisticRegressor(num_feats=3, learning_rate=0.012, tol=0.01, max_iter=10, batch_size=10)
	log_model.W = np.array([1, -2, 5, -6])
	X = np.array([[1, 1, 2], [3, 5, 8], [34, 21, 13]])
	X = np.hstack([X, np.ones((X.shape[0], 1))])
	y_true = np.array([3, 27, 51])
	y_true = 1 / (1 + np.exp(y_true))
	y_pred = log_model.make_prediction(X)
	assert np.all(np.abs(y_pred - y_true) < 0.00001)


def test_loss_function():
	# Test the loss function on a small dummy example
	log_model = LogisticRegressor(num_feats=3, learning_rate=0.012, tol=0.01,max_iter=10, batch_size=10)
	log_model.W = np.array([1, -2, 5, -6])
	X = np.array([[1, 1, 2], [3, 5, 8], [34, 21, 13]])
	X = np.hstack([X, np.ones((X.shape[0], 1))])
	y_true = np.array([4, 25, 50])
	y_true = 1 / (1 + np.exp(y_true))
	y_pred = log_model.make_prediction(X)
	loss_result = log_model.loss_function(y_true, y_pred)
	true_loss = 0.03418199394562356
	assert (np.abs(loss_result - true_loss) < 0.0001)

def test_gradient():
	# Test the gradient on a small dummy example
	log_model = LogisticRegressor(num_feats=3, learning_rate=0.012, tol=0.01,max_iter=10, batch_size=10)
	log_model.W = np.array([1, -2, 5, -6])
	X = np.array([[1, 1, 2], [3, 5, 8], [34, 21, 13]])
	X = np.hstack([X, np.ones((X.shape[0], 1))])
	y_true = np.array([4, 25, 50])
	y_true = 1 / (1 + np.exp(y_true))
	gradient_result = log_model.calculate_gradient(y_true, X)
	gradient_true = np.array([-0.02943966, -0.02943966, -0.05887933, -0.02943966])
	assert (np.all(np.abs(gradient_result - gradient_true) < 0.0001))
	

def test_training():
	# Test training on a small dummy example
	log_model = LogisticRegressor(num_feats=3, learning_rate=0.012, tol=0.01,max_iter=3, batch_size=2)

	X_train = np.array([[2, 2, 4], [3, 5, 10], [1, 1, 3], [5, 2, 7]])
	y_train = np.array([1, 0, 0, 1])
	y_train = 1 / (1 + np.exp(y_train))

	X_val = np.array([[3, 2, 4], [3, 5, 8]])
	y_val = np.array([0, 1])
	y_val = 1 / (1 + np.exp(y_val))

	W_initial = np.copy(log_model.W)
	log_model.train_model(X_train, y_train, X_val, y_val)
	W_final = np.copy(log_model.W)
	# Assert that the weights are updated
	# We cannot guarantee the loss or error rate decreases on update
	assert np.all(W_initial != W_final)
