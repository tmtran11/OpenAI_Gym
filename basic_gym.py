# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 09:51:13 2018

@author: FPTShop
"""

import gym
from gym import wrappers
import numpy as np
from matplotlib import pyplot as plt

# play one episode, return length
# play multiple episode, with params, return avg length
# choose param use make average length
# a action decision, take params
# a main game
n = np.linspace(0,9)
def take_action(s, p):
    return 1 if s.dot(p)>0 else 0
def play_one_ep(evn, p):
    s = env.reset()
    done = False
    t = 0
    while not done and t<200:
        t += 1
        action = take_action(s,p)
        s, reward, done, info = env.step(action)
        if done: break
    return t
def play_mul_ep(env, p, T):
    ep_length = []
    for i in range(T):
        ep_length.append(play_one_ep(env, p))
    return sum(ep_length)/float(len(ep_length))
def choose_param(env, n, t):
    longest_ep = 0.0
    best_p = np.random.random(4)*2/1
    for i in range(n):
        print(i)
        p = np.random.random(4)*2/1
        ep_length = play_mul_ep(env, p, t)
        if ep_length>longest_ep:
            longest_ep = ep_length
            best_p = p
    return best_p

env = gym.make('CartPole-v0')
choose_param(env, 100, 100)