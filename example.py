#!/usr/bin/env python
#coding:utf-8

from hmm import Model

#The hidden states 
states = ['rainy','sunny','cloudy']

#The observation states
observation = ['walk','shop','clean']


#The intial probability for the hidden states
phi = {'rainy':0.333,'sunny':0.333,'cloudy':0.333}

#The trans prob for the hidden states
trans_prob = {
    'rainy':{'rainy':0.4,'sunny':0.3,'cloudy':0.3},
    'sunny':{'rainy':0.3,'sunny':0.4,'cloudy':0.3},
    'cloudy':{'rainy':0.3,'sunny':0.4,'cloudy':0.3}
}

#The prob of observation in condition of a hidden state
conf_prob = {
    'rainy':{'walk':0.1,'shop':0.3,'clean':0.6},
    'sunny':{'walk':0.4,'shop':0.5,'clean':0.1},
    'cloudy':{'walk':0.6,'shop':0.25,'clean':0.15}
}

observations = ['walk', 'shop', 'clean', 'clean', 'walk', 'walk']

model = Model(states,observation,phi,trans_prob,conf_prob)

print model.evaluate(observations)
print model.decode(observations)
