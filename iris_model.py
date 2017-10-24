import tensorflow as tf
import numpy as np
import pandas as pd

class iris_bank:
    def __init__(self, size_train, size_valid):
        df = pd.read_csv("iris.txt", "\t")
        matrix = np.array(df.values[:, :])
        np.random.shuffle(matrix)
        matrix_split = np.split(matrix, [matrix.shape[1] - 1, matrix.shape[1]], axis=1)
        matrix_feature = matrix_split[0]
        matrix_class = matrix_split[1]
        matrix_blank = []
        self.T, self.V, self.S = {}, {}, {}
        for row in range(matrix_class.shape[0]):
            if 1. == matrix_class[row, 0]:
                matrix_blank.append([0, 0, 1])
            elif 2. == matrix_class[row, 0]:
                matrix_blank.append([0, 1, 0])
            else:
                matrix_blank.append([1, 0, 0])
        matrix_class = np.array(matrix_blank)
        result1 = np.split(matrix_feature, [size_train, size_train + size_valid, matrix.shape[0]], axis=0)
        result2 = np.split(matrix_class, [size_train, size_train + size_valid, matrix.shape[0]], axis=0)
        self.T['fea'], self.V['fea'], self.S['fea'] = result1[0], result1[1], result1[2]
        self.T['cla'], self.V['cla'], self.S['cla'] = result2[0], result2[1], result2[2]
        self.features = matrix_feature.shape[1]

class iris_seeker:
    def __init__(self, dataset, dropout):
        with tf.name_scope("Basic_Settings"):
            self.train_dict, self.valid_dict, self.test_dict = {}, {}, {}
            self.weights, self.bias = {}, {}
            self.data = dataset
            self.drop = dropout
            self.stack = 0

    def input_layer(self):
        with tf.name_scope("Input_Layer"):
            self.input = tf.placeholder("float", [None, self.data.features])
            self.train_dict[self.input] = self.data.T['fea']
            self.valid_dict[self.input] = self.data.V['fea']
            self.test_dict[self.input] = self.data.S['fea']
        return self.input

    def single_hidden_layer(self, target, dim):
        with tf.name_scope("Hidden_Layer"):
            weights = tf.get_variable(name="W_input2hidden", shape=[self.data.features, dim],
                                      initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable(name="B_input2hidden", shape=[dim],
                                   initializer=tf.contrib.layers.xavier_initializer())
            result = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(target, weights), bias)), self.drop)
        return result

    def multi_hidden_layer(self, target, dim0, dim1):
        self.stack += 1
        with tf.name_scope("Hidden_Layer_" + str(self.stack)):
            weights = tf.get_variable(name="W_hidden_" + str(self.stack),
                                      shape=[dim0, dim1], initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable(name="B_hidden_" + str(self.stack),
                                   shape=[dim1], initializer=tf.contrib.layers.xavier_initializer())
            result = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(target, weights), bias)), self.drop)
        return result

    def output_layer(self, target, dim):
        with tf.name_scope("Output_Layer_" + str(self.stack)):
            self.answer = tf.placeholder("float", [None, 3])
            self.train_dict[self.answer] = self.data.T['cla']
            self.valid_dict[self.answer] = self.data.V['cla']
            self.test_dict[self.answer] = self.data.S['cla']
            weights = tf.get_variable(name="W_hidden2output",
                                      shape=[dim, 3], initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable(name="B_hidden2output",
                                   shape=[3], initializer=tf.contrib.layers.xavier_initializer())
            result = tf.nn.dropout(tf.nn.sigmoid(tf.add(tf.matmul(target, weights), bias)), self.drop)
        return result

    def optimize(self, result, epochs, learn):
        with tf.name_scope("Optimization"):
            cost = tf.reduce_mean(tf.pow(result - self.answer, 2))
            opti = tf.train.GradientDescentOptimizer(learn).minimize(cost)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            with tf.Session(config=config) as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                for iter in range(epochs):
                    train_cost, _ = sess.run([cost, opti], feed_dict=self.train_dict)
                    valid_cost = sess.run(cost, feed_dict=self.valid_dict)
                    if iter % 1000 == 0:
                        print("Iteration: ", iter)
                        print("Cost for training session: ", train_cost)
                        print("Cost for validation session: ", valid_cost)
                test_cost = sess.run(cost, feed_dict=self.test_dict)
                print("Final training cost: ", train_cost,
                      "Final validation cost: ", valid_cost,
                      "Final test cost: ", test_cost)
            sess.close()

