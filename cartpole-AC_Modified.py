
from __future__ import print_function, division
from builtins import range
import gym
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from utilss import FeatureTransformer, plot_rewards_running_avg,HiddenLayer




# approximates pi(a | s)
class PolicyModel:
  def __init__(self, D, K, hidden_layer_sizes):
    # create the graph
    # K = number of actions
    self.layers = []
    M1 = D
    for M2 in hidden_layer_sizes:
      layer = HiddenLayer(M1, M2)
      self.layers.append(layer)
      M1 = M2

    # final layer
    # layer = HiddenLayer(M1, K, lambda x: x, use_bias=False)
    layer = HiddenLayer(M1, K, tf.nn.softmax, use_bias=False)
    self.layers.append(layer)

    # inputs and targets
    self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
    self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')
    self.old_p_a_given_s=1
    
    
    
    Z = self.X
    for layer in self.layers:
      Z = layer.forward(Z)
    self.p_a_given_s = Z
    
    ############ calculate output and cost  ###########
    
    self.predict_op = self.p_a_given_s
    ch_p_a_given_s=self.p_a_given_s/self.old_p_a_given_s
    
    selected_probs = tf.log(
      tf.reduce_sum(
        ch_p_a_given_s * tf.one_hot(self.actions, K),
        reduction_indices=[1]
      )
    )
    cost = -tf.reduce_sum(self.advantages * selected_probs)
    self.train_op = tf.train.AdagradOptimizer(1e-1).minimize(cost)
    
    

  def set_session(self, session):
    self.session = session
  def update_old_p_a_given_s(self):
    p =self.old_p_a_given_s
    self.old_p_a_given_s = self.p_a_given_s
    return(p,self.old_p_a_given_s)

  def partial_fit(self, X, actions, advantages):
    X = np.atleast_2d(X)
    actions = np.atleast_1d(actions)
    advantages = np.atleast_1d(advantages)
    self.session.run(
      self.train_op,
      feed_dict={
        self.X: X,
        self.actions: actions,
        self.advantages: advantages,
      }
    )

  def predict(self, X):
    X = np.atleast_2d(X)
    return self.session.run(self.predict_op, feed_dict={self.X: X})

  def sample_action(self, X):
    p = self.predict(X)[0]
    return np.random.choice(len(p), p=p)


# approximates V(s)
class ValueModel:
  def __init__(self, D, hidden_layer_sizes):
    # create the graph
    self.layers = []
    M1 = D
    for M2 in hidden_layer_sizes:
      layer = HiddenLayer(M1, M2)
      self.layers.append(layer)
      M1 = M2

    # final layer
    layer = HiddenLayer(M1, 1, lambda x: x)
    self.layers.append(layer)

    # inputs and targets
    self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
    self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

    # calculate output and cost
    Z = self.X
    for layer in self.layers:
      Z = layer.forward(Z)
    Y_hat = tf.reshape(Z, [-1]) # the output
    self.predict_op = Y_hat

    cost = tf.reduce_sum(tf.square(self.Y - Y_hat))
    #self.train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)
    # self.train_op = tf.train.MomentumOptimizer(1e-2, momentum=0.9).minimize(cost)
    self.train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(cost)

  def set_session(self, session):
    self.session = session

  def partial_fit(self, X, Y):
    X = np.atleast_2d(X)
    Y = np.atleast_1d(Y)
    self.session.run(self.train_op, feed_dict={self.X: X, self.Y: Y})

  def predict(self, X):
    X = np.atleast_2d(X)
    return self.session.run(self.predict_op, feed_dict={self.X: X})


def play_one_td(env, pmodel, vmodel, gamma):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0
  while not done and iters < 2000:
    # if we reach 2000, just quit, don't want this going forever
    # the 200 limit seems a bit early
    action = pmodel.sample_action(observation)
    prev_observation = observation
    observation, reward, done, info = env.step(action)

    # if done:
    #   reward = -200

    # update the models
    V_next = vmodel.predict(observation)[0]
    G = reward + gamma*V_next
    advantage = G - vmodel.predict(prev_observation)
    pmodel.partial_fit(prev_observation, action, advantage)
    vmodel.partial_fit(prev_observation, G)
    
    if reward == 1: # if we changed the reward to -200
      totalreward += reward
    iters += 1

  return totalreward



def main():
  env = gym.make('CartPole-v0')
  D = env.observation_space.shape[0]
  K = env.action_space.n
  pmodel = PolicyModel(D, K, [])
  vmodel = ValueModel(D, [10])
  init = tf.global_variables_initializer()
  session = tf.InteractiveSession()
  session.run(init)
  pmodel.set_session(session)
  vmodel.set_session(session)
  gamma = 0.99

  if 'monitor' in sys.argv:
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)

  N = 2000
  totalrewards = np.empty(N)
  costs = np.empty(N)
  for n in range(N):
    totalreward = play_one_td(env, pmodel, vmodel, gamma)
    totalrewards[n] = totalreward
    if n % 100 == 0:
      print("episode:", n, "total reward:", totalreward, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean(),"SD (last 100) :" ,totalrewards[max(0, n-100):(n+1)].std())

  print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
  print("SD :", totalrewards.std())
  print("Mean :", totalrewards.mean())
  print("CV :",totalrewards.std()/totalrewards.mean())
  

  plot_rewards_running_avg(totalrewards)


if __name__ == '__main__':
  main()

