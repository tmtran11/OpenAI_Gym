# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 13:06:31 2018
Cart-pole with learning
@author: FPTShop
"""

import os
import gym
import numpy as np
import matplotlib.pyplot as plt

# def toint
# class ft_transform: initialize 4 bin, transform method and then toint.
# class model: initialize numpy Q[s][a], 2 action, ? state, take_action, td_update, states.
# def play_one_eposode(env,lr,eps): call current state, run expected G, update
# def play multiple of time, with epsilon decrease
# plot time play each episodes?

def toint(features):
    return int(''.join(map(lambda feature: str(int(feature)), features)))

def tobin(var,bins):
    return np.digitize([var],bins)

class Fit_transform:
    def __init__(self):
        self.cart_position_bins = np.linspace(-2.4,2.4,9)
        self.cart_velocity_bins = np.linspace(-10,10,9)
        self.pole_angle_bins = np.linspace(-0.41,0.41,9)
        self.pole_velocity_bins = np.linspace(-3.5,3.5,9)
    def transform(self,state):
        cart_position, cart_velocity, pole_angle, pole_velocity = state
        return toint([tobin(cart_position, self.cart_position_bins), 
                      tobin(cart_velocity,self.cart_velocity_bins),
                      tobin(pole_angle,self.pole_angle_bins),
                      tobin(pole_velocity,self.pole_velocity_bins)])
class Model():
    def __init__(self, fit_transform):
        self.Q = np.random.uniform(-1.0, 1.0, (10**4,2))
        self.fit_transform = fit_transform
    def predict(self, state):
        s = self.fit_transform.transform(state)
        return self.Q[s]
    def take_action(self, env, state, eps):
        p = np.random.random()
        if p<eps:
            return env.action_space.sample()
        else:
            value = self.predict(state)
            return np.argmax(value)
    def td_update(self, G, prev_state, action):
        td = 0.01
        prev_state = self.fit_transform.transform(prev_state)
        self.Q[prev_state, action] = self.Q[prev_state, action] + td*(G-self.Q[prev_state, action])
        
def play_one_ep(model, env, eps, lr):
    state = env.reset()
    done = False
    all_reward = 0
    t = 0
    while not done and t<100000:
        t += 1
        action = model.take_action(env, state, eps)
        prev_state = state
        state, reward, done, info = env.step(action)
        value = model.predict(state)
        if done and t < 200:
            reward = -300
        G = reward + lr*np.max(value)
        model.td_update(G,prev_state,action)
        all_reward += reward
    return t

def train(N, eps, lr):
    env = gym.make('CartPole-v0')
    fit_transform = Fit_transform()
    model = Model(fit_transform)
    all_reward = []
    for x in range(N):
        print(x)
        eps = 1.0/np.sqrt(x+1)
        all_reward.append(play_one_ep(model, env,  eps, lr))
    plt.plot(all_reward)
    plt.show()
    
train(30000, 0.05, 0.9)

