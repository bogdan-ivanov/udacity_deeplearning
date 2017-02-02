# coding: utf-8
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data_path = 'Bike-Sharing-Dataset/hour.csv'

rides = pd.read_csv(data_path)
rides.head()
# rides[:24*10].plot(x='dteday', y='cnt')


# Here we have some categorical variables like season, weather, month. To include these in our model, we'll need to make binary dummy variables. This is simple to do with Pandas thanks to `get_dummies()`.

dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
data.head()


# To make training the network easier, we'll standardize each of the continuous variables. That is, we'll shift and scale the variables such that they have zero mean and a standard deviation of 1.
quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std


test_data = data[-21*24:]
data = data[:-21*24]

target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes**-0.5, 
                                       (self.output_nodes, self.hidden_nodes))
        self.lr = learning_rate
        
        #### Set this to your implemented sigmoid function ####
        # Activation function is the sigmoid function
        self.activation_function = lambda x: 1.0 / (1 + np.exp(-x))
    
    def train(self, inputs_list, targets_list):
        """
        inputs_list = 1D list of inputs
        targets_list = 1D list of outputs
        """
        x = np.array(inputs_list)
        y = np.array(targets_list)
        
        hidden_outputs, final_outputs = self.forward(x)

        output_errors = y - final_outputs # Output layer error is the difference between desired target and actual output.
        output_grad = output_errors


        # TODO: Backpropagated error
        hidden_errors = np.dot(output_errors, self.weights_hidden_to_output) * hidden_outputs * (1 - hidden_outputs) # errors propagated to the hidden layer

        # TODO: Update the weights
        self.weights_hidden_to_output += self.lr * output_grad * hidden_outputs # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * hidden_errors.T[:, None] * x[:, None].T # update input-to-hidden weights with gradient descent step

    def run(self, inputs_list):
        """ inputs_list = 1-D list of inputs """
        outputs = []
        for inputs in inputs_list.as_matrix():
            # Run a forward pass through the network
            inputs = np.array(inputs)
            hidden_outputs, final_outputs = self.forward(np.array(inputs))
            outputs.append(final_outputs)

        return np.array(outputs)

    def forward(self, x):
        """ x = 1D-input vector"""
        
        hidden_inputs = self.weights_input_to_hidden.dot(x)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = self.weights_hidden_to_output.dot(hidden_outputs)
        final_outputs = final_inputs

        return hidden_outputs, final_outputs
        

def MSE(y, Y):
    return np.mean((y-Y)**2)




### Set the hyperparameters here ###
epochs = 100
learning_rate = 0.005
hidden_nodes = 500
output_nodes = 1

N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train': [], 'validation': []}
for e in range(epochs):
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(train_features.index, size=128)
    for record, target in zip(train_features.ix[batch].values, 
                              train_targets.ix[batch]['cnt']):
        network.train(record, target)
    
    # Printing out the training progress
    train_loss = MSE(network.run(train_features), train_targets['cnt'].values)
    val_loss = MSE(network.run(val_features), val_targets['cnt'].values)
    sys.stdout.write("\rProgress: " + str(100 * e/float(epochs))[:4] +
                     "% ... Training loss: " + str(train_loss)[:5] +
                     " ... Validation loss: " + str(val_loss)[:5])
    
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)

#
# print losses['train']
# print losses['validation']
plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
# # plt.ylim(ymax=0.5)
plt.show()
#

# # ## Check out your predictions
# #
# # Here, use the test data to view how well your network is modeling the data. If something is completely wrong here, make sure each step in your network is implemented correctly.
#
#
fig, ax = plt.subplots(figsize=(8,4))

mean, std = scaled_features['cnt']
predictions = network.run(test_features)*std + mean

print "predictions=", predictions

ax.plot(predictions[:, 0], label='Prediction')
ax.plot((test_targets['cnt']*std + mean).values, label='Data')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)
plt.show()


# import unittest
#
# inputs = [0.5, -0.2, 0.1]
# targets = [0.4]
# test_w_i_h = np.array([[0.1, 0.4, -0.3],
#                        [-0.2, 0.5, 0.2]])
# test_w_h_o = np.array([[0.3, -0.1]])
#
# class TestMethods(unittest.TestCase):
#
#     ##########
#     # Unit tests for data loading
#     ##########
#
#     def test_data_path(self):
#         # Test that file path to dataset has been unaltered
#         self.assertTrue(data_path.lower() == 'bike-sharing-dataset/hour.csv')
#
#     def test_data_loaded(self):
#         # Test that data frame loaded
#         self.assertTrue(isinstance(rides, pd.DataFrame))
#
#     ##########
#     # Unit tests for network functionality
#     ##########
#
#     def test_activation(self):
#         network = NeuralNetwork(3, 2, 1, 0.5)
#         # Test that the activation function is a sigmoid
#         self.assertTrue(np.all(network.activation_function(0.5) == 1/(1+np.exp(-0.5))))
#
#     def test_train(self):
#         # Test that weights are updated correctly on training
#         network = NeuralNetwork(3, 2, 1, 0.5)
#         network.weights_input_to_hidden = test_w_i_h.copy()
#         network.weights_hidden_to_output = test_w_h_o.copy()
#
#         network.train(inputs, targets)
#         self.assertTrue(np.allclose(network.weights_hidden_to_output,
#                                     np.array([[ 0.37275328, -0.03172939]])))
#         self.assertTrue(np.allclose(network.weights_input_to_hidden,
#                                     np.array([[ 0.10562014,  0.39775194, -0.29887597],
#                                               [-0.20185996,  0.50074398,  0.19962801]])))
#
#     def test_run(self):
#         # Test correctness of run method
#         network = NeuralNetwork(3, 2, 1, 0.5)
#         network.weights_input_to_hidden = test_w_i_h.copy()
#         network.weights_hidden_to_output = test_w_h_o.copy()
#
#         self.assertTrue(np.allclose(network.run(inputs), 0.09998924))
#
# suite = unittest.TestLoader().loadTestsFromModule(TestMethods())
# unittest.TextTestRunner().run(suite)

