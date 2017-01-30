import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt





if __name__ == "__main__":

    # Read the data
    df = pd.read_csv('data.csv')

    # Drop features
    df = df.drop(['index', 'price', 'sq_price'], axis=1)

    # Only use the first 10 rows
    df = df[:10]

    # print df

    # Add labels
    df.loc[:, ('y1')] = [1, 1, 1, 0, 0, 1, 0, 1, 1, 1]

    # y2 is the negation of y1
    df.loc[:, ('y2')] = df['y1'] == 0
    df.loc[:, ('y2')] = df['y2'].astype(int)

    print df

    X_train = df.loc[:, ['area', 'bathrooms']].as_matrix()
    Y_train = df.loc[:, ['y1', 'y2']].as_matrix()

    print X_train, Y_train

    LEARNING_RATE = 0.000001
    TRAINING_EPOCHS = 2000
    DISPLAY_STEP = 50
    SAMPLE_COUNT = len(X_train)

    # None = Any number of samples
    # 2 = the number of features

    # placeholders = gateways for data into our computation graph
    x = tf.placeholder(tf.float32, [None, 2])

    # 2x2 float matrix - keep updating through the training process
    W = tf.Variable(tf.zeros([2, 2]))

    # add biases
    b = tf.Variable(tf.zeros([2]))

    y_values = tf.add(tf.matmul(x, W), b)
    y = tf.nn.softmax(y_values)

    y_ = tf.placeholder(tf.float32, [None, 2])

    cost = tf.reduce_sum(tf.pow(y_ - y, 2)) / (2 * SAMPLE_COUNT)

    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)


    # Initialize variables and tensorflow session
    init = tf.initialize_all_variables()

    session = tf.Session()
    session.run(init)

    for i in range(TRAINING_EPOCHS):
        session.run(optimizer, feed_dict={x: X_train, y_: Y_train})

        if not i % DISPLAY_STEP:
            cc = session.run(cost, feed_dict={x: X_train, y_: Y_train})

            print "Training step: ", i, 'cost=', cc

    print "Optimization finished"
    training_cost = session.run(cost, feed_dict={x: X_train, y_: Y_train})
    print "Training cost=", training_cost, "W=", session.run(W), "b=", session.run(b)

    print session.run(y, feed_dict={x: X_train})



