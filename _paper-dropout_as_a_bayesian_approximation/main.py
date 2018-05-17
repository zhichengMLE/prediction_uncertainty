# -*- coding: utf-8 -*-
"""
Created on Wed May 16 14:38:37 2018

@author: Administrator
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import Bernoulli
import matplotlib.pyplot as plt

class VariationalDense():
    def __init__(self, input_data, output_data, model_prob, model_lam):
        self.model_prob = model_prob
        self.model_lam = model_lam
        self.model_bern = Bernoulli(probs=self.model_prob, dtype=tf.float32)
        self.model_M = tf.Variable(tf.truncated_normal((input_data, output_data), stddev=0.01))
        self.model_m = tf.Variable(tf.zeros((output_data)))
        self.model_W = tf.matmul(tf.diag(self.model_bern.sample((input_data,))), self.model_M)
        
    
    def __call__(self, X_train, activation=tf.identity):
        output = activation(tf.matmul(X_train, self.model_W) + self.model_m)
        
        if self.model_M.shape[1] == 1:
            output = tf.squeeze(output)
            
        return output
    
    @property
    def regularization(self):
        return self.model_lam * (
            self.model_prob * tf.reduce_sum(tf.square(self.model_M)) + tf.reduce_sum(tf.square(self.model_m))
        )

if __name__ == '__main__':
    # Generate demo data
    n_sample = 2000
    X_train = np.random.normal(size=(n_sample, 1))
    y_train = np.random.normal(np.cos(5. * X_train) / (np.abs(X_train) + 1.), 0.1).ravel()
    X_pred = np.atleast_2d(np.linspace(-3., 3., num=100)).T
    
    X_train = np.hstack((X_train, X_train**2, X_train**3))
    X_pred = np.hstack((X_pred, X_pred**2, X_pred**3))

    # Create model
    n_features = X_train.shape[1]
    n_hidden = 100
    model_prob = 0.95
    model_lam = 1e-2
    model_X = tf.placeholder(tf.float32, [None, n_features])
    model_y = tf.placeholder(tf.float32, [None])
    model_L_1 = VariationalDense(n_features, n_hidden, model_prob, model_lam)
    model_L_2 = VariationalDense(n_hidden, n_hidden, model_prob, model_lam)
    model_L_3 = VariationalDense(n_hidden, n_hidden, model_prob, model_lam)
    model_L_4 = VariationalDense(n_hidden, 1, model_prob, model_lam)
    
    model_out_1 = model_L_1(model_X, tf.nn.relu)
    model_out_2 = model_L_2(model_out_1, tf.nn.relu)
    model_out_3 = model_L_3(model_out_2, tf.nn.relu)
    model_pred = model_L_4(model_out_3)
    model_sse = tf.reduce_sum(tf.square(model_y - model_pred))
    model_mse = model_sse / n_sample
    model_loss = (
        model_sse + model_L_1.regularization + model_L_2.regularization + 
        model_L_3.regularization + model_L_4.regularization
    ) / n_sample

    train_step = tf.train.AdamOptimizer(1e-3).minimize(model_loss)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            sess.run(train_step, {model_X: X_train, model_y: y_train})
            if(i % 100 == 0):
                mse = sess.run(model_mse, {model_X: X_train, model_y: y_train})
                print("Iteration: %d, MSE: %.4f" %(i, mse))
                
        
        # Sample from the posterior
        n_post = 1000
        y_post = np.zeros((n_post, X_pred.shape[0]))
        for i in range(n_post):
            y_post[i] = sess.run(model_pred, {model_X: X_pred})

    # Plot dataset
    plt.figure(figsize=(20, 10))
    for i in range(n_post):
        plt.plot(X_pred[:, 0], y_post[i], "b-", alpha=1. / 100)
    plt.plot(X_train[:, 0], y_train, "r.")
    plt.grid()
    plt.show()