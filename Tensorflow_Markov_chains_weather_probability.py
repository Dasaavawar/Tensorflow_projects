#Markov Chains model for Tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Import libraries
import tensorflow_probability as tfp
import tensorflow as tf

#Select type of data and write probability graphs
tfd = tfp.distributions
initial_distribution = tfd.Categorical(probs=[0.8, 0.2])
transition_distribution = tfd.Categorical(probs=[[0.7, 0.3], [0.2, 0.8]])
observation_distribution = tfd.Normal(loc=[0.0, 15.0], scale=[5.0, 10.0])

#Create model
model = tfd.HiddenMarkovModel(
    initial_distribution = initial_distribution,
    transition_distribution = transition_distribution,
    observation_distribution = observation_distribution,
    num_steps = 7
)

mean = model.mean()

with tf.compat.v1.Session() as sess:
    print(mean.numpy())