#!/usr/bin/env python
#coding:utf-8

from hmm import Model

#The hidden states 
states = [1,2,3]

#The observation states
observation = [1,2]


#The intial probability for the hidden states
phi = {1:0.333,2:0.333,3:0.333}

#The trans prob for the hidden states
trans_prob = {
    1:{1:0.333,2:0.333,3:0.333},
    2:{1:0.333,2:0.333,3:0.333},
    3:{1:0.333,2:0.333,3:0.333}
}

#The prob of observation in condition of a hidden state
conf_prob = {
    1:{1:0.5,2:0.5},
    2:{1:0.75,2:0.25},
    3:{1:0.25,2:0.75}
}

observations =[1,1,1,1,2,1,2,2,2,2]

model = Model(states,observation,phi,trans_prob,conf_prob)

print model.evaluate(observations)
print model.decode(observations)
