#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 18:55:05 2019

@author: hyeon

AI for Self Driving Car
"""

# Importing the libraries
import numpy as np
import random  # 랜덤한 인풋값으로 테스트하기 위해서.
import os  # for saving and loading the results.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  # for optimizer
import torch.autograd as autograd
from torch.auto import Variable


# Creating the architecutre fo the Neural Network
# init func & forward func을 가지는 class 하나를 만들것임.
# init func : input, hidden, output layer들을 만듦.
# forward func : activate the neurons in the neural network (Rectified activation function)

class Network(nn.Module):
    
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()  # nn.Module 의 모든것을 사용할 수 있게 됨.
        self.input_size = input_size  # input layer에서 input neuron의 개수
        self.nb_action = nb_action  # output layer에서 output의 개수
        
        # 2개의 full connection 이 필요함. 1: input layer <-> hidden layer / 2: hidden layer <-> output layer
        # nn.Linear()이 full connection을 만들어줌.
        self.fc1 = nn.Linear(input_size, 30) # 5개의 input neurons과 30개의 hidden neurons의 full connection을 생성.
        self.fc2 = nn.Linear(30, nb_action) # 30개의 hidden neurons과 3개의 output neurons사이의 full connection을 생성.
        
    # performing for propagation
    # output Q-values (3 possible actions: 1.go left, 2.go straight, 3.go right)
    def forward(self, state):
        # activate the hidden neuron (x is the activated hidden neuron) by using ReLU func.
        # 모양이 대략 state -> x -> q_values 이런식.
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values