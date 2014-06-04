#!/usr/bin/env python
#coding:utf-8

class Model(object):
    def __init__(self,states,observation,phi,trans_prob,conf_prob):
        self._states = states
        self._observation = observation
        self._phi = phi
        self._trans_prob = trans_prob
        self._conf_prob = conf_prob
        
    def states_length(self):
        #Return the length of the states
        return len(self._states)

    def _forward(self,observations):
        #The implemention of the forward algorithm
        s_len = self.states_length
        o_len = len(observations)
        '''
        This step should cal the alpha_t(j)
        the t is the length of the observations,
        the j is the hidden states
        '''
        alpha = [[] for i in range(o_len)]
        
        alpha[0] = {}
        #t=1,cal the intil alpha_1(j)
        for state in self._states:
            alpha[0][state] = self._conf_prob[state][observations[0]]*self._phi[state]
        
        #t>1,cal the local prob alpha_t(j)
        for index in range(1,o_len):
            alpha[index] ={}
            for state_to in self._states:
                #the time t the prob all path that direct to states_to
                prob = 0
                for state_from in self._states:
                    prob += alpha[index-1][state_from]*self._trans_prob[state_from][state_to]
                alpha[index][state_to]=self._conf_prob[state_to][observations[index]]*prob
        return alpha
        
    def _viterbi(self,observations):
        #The implemention of the viterbi algorithm
        s_len = self.states_length
        o_len = len(observations)
        '''
        This step should cal the beta_t(j),
        the t is the length of the observations,
        the j is the hidden states,
        the beta_t(j) means at time t the most probable 
        local path to state j
        '''
        beta = [[] for i in range(o_len)]
        beta[0] = {}
        
        for state in self._states:
            beta[0][state] = self._conf_prob[state][observations[0]]*self._phi[state]
            
        #t>1,cal the local prob beta_t(j)
        for index in range(1,o_len):
            beta[index] = {}
            for state_to in self._states:
                #build a list to save the beta_t-1(j)a_jib_ikt
                prob = []
                for state_from in self._states:
                    temp = beta[index-1][state_from]*self._trans_prob[state_from][state_to]*self._conf_prob[state_to][observations[index]]
                    prob.append(temp)
                prob =sorted(prob,reverse = True)
                beta[index][state_to] = prob[0]

        return beta
    
    def _backward_point(self,beta,observations,state):
        """
        rely on the beta to get the state sequences that best 
        explain the observation sequences
        """
        index = len(observations)-1
        theta =[0 for i in range(len(observations))]
        theta[index] = state
        while index >0:
            prob = {}
            for state_from in self._states:
                prob[state_from] = beta[index-1][state_from]*self._trans_prob[state_from][state]
            state = sorted(prob,key=prob.get,reverse=True)[0]
            index -= 1
            theta[index] = state
        return theta
        
    def _inverse(self,beta):
        result = [0 for i in range(len(beta))] 
        length = len(beta)
        for i in range(len(beta)):
            result[i] = beta[length-i-1]
        return result

    def evaluate(self,observations):
        """
        use the forward algorithm to cal the 
        prob of the observation sequence under the HMM Model
        """
        length = len(observations)
        if length == 0:
            return 0
        
        alpha = self._forward(observations)
        prob = sum(alpha[length-1].values())
        return prob
        
    def decode(self,observations):
        """
        user the be viterbi algorithm to cal the most probable 
        hidden state sequence to the observations sequence ,
        """
        length = len(observations)
        if length == 0 :
            return 0
        beta = self._viterbi(observations)
        #get the last state to the last obseravtions
        sequence = beta[length-1]
        state = sorted(sequence,key=sequence.get,reverse=True)[0]
        theta = self._backward_point(beta,observations,state)
        return theta
