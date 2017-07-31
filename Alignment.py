#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 09:36:52 2017

@author: rohit
"""


import tensorflow as tf
import tensorflow.contrib.slim as slim
import ubmatrix
import numpy as np
import time

sess = tf.InteractiveSession()

first_peak = []


def finished(s):
    """
    determines if the crystal is in a diffracting position for the 100 equivalences
    """   
    s = [s[0], s[1]]
    x = ubmatrix.motor_pos([1,0,0])                 #list of motor positions for 100 equivalences
    if (s in x) :   
        print("--------------------DONE--------------------", x[x.index(s)])
        global first_peak
        first_peak = x[x.index(s)]
        return True
    return False

def finished2(s):
    """
    determines if the crystal is in a different diffracting position for the 100 equivalences
    NOT YET IMPLEMENTED
    """
    s = [s[0], s[1]]
    x = ubmatrix.motor_pos([1,0,0])
    if (s in x) and not (x[x.index(s)] == first_peak or x[x.index(s)] == np.negative(first_peak)):
        print("--------------------DONE--------------------", x[x.index(s)])
        return True
    return False    


def action(a):
    """
    Returns the action desired by index 'a'
    the action list could contain negative movements as well but for now it is only positive.
    """
    return [[1,0],[0,1],[1,1]][a]

def reward(s):
    """ 
    rewards the agent if the crystal is in diffracting postion
    reward == 1 if hit, 0 otherwise
    """

    if finished(s):
        return 1
    return 0


def discounted_rewards(r):
    """
    increases the reward for moves that result in a reflection based on their proximity to the refleciton.
    moves right before a reflection is hit gets a larger reward than moves 
    """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in (range(0, np.size(r))):
        running_add = running_add*0.99 + r[t]
        discounted_r[t] = running_add
    return discounted_r


class agent():
    """
    Machine learning agent. uses the motor positions and previous experiences to make the next move. After every reflection is hit, the agent
    updastes its network to speed up the reflection finding process. Uses the Adam Optimizer provided by tensorflow.
    """
    def __init__(self, lr, s_size, a_size, h_size):
        self.state_in = tf.placeholder(shape = [None, s_size], dtype=tf.float32, name = "STATE_IN")
        hidden = slim.fully_connected(self.state_in, h_size, biases_initializer=None, activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden, a_size, biases_initializer=None, activation_fn = tf.nn.softmax)
        
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32, name = "REWARD_HOLDER")
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32, name= "ACTION_HOLDER")

        self.indexes = tf.range(0, tf.shape(self.output)[0]) *tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs= tf.gather(tf.reshape(self.output, [-1]), self.indexes)  #the probabilities that were chosen
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder[-1])    # the mean of the discounted reward times ln(responsible outputs)

        tvars = tf.trainable_variables()                # The hidden net and the output layer
        self.gradient_holders = []
        for idx, var in enumerate(tvars):               #idx = index, var = layer
            placeholder = tf.placeholder(tf.float32, name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        self.gradients = tf.gradients(self.loss, tvars)  
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)  #the method of optimizing we use
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))
        
tf.reset_default_graph()
merged = tf.summary.merge_all()
myAgent = agent(lr = 0.0001, s_size = 2, a_size = 3, h_size = 400)  # learning rate, state size, action size, hidden node size


# iterations of episodes, max number of movements
total_episodes = 5000
max_ep = 100000

update_freqency = 1

# initialize the agent
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    i=0
    s=[0,0]
    total_reward = []
    total_length = []
    total = []
    gradBuffer = sess.run(tf.trainable_variables())
    
    
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad*0
    
    while i < total_episodes:               #begin training agent  
        running_reward = 0
        ep_history = []
        
        for j in range(max_ep):
            s = [s[0]%360, s[1]%360]
            a_dist = sess.run(myAgent.output, feed_dict = {myAgent.state_in:[s]})           #action distribution from agent
            
            a = np.random.choice(a_dist[0], p = a_dist[0])
            l1 = []
            for n in range(len(a_dist[0])):
                if (a+0.01 > a_dist[0][n] > a-0.01):
                    l1.append(n)
            a = np.random.choice(l1)                                                        #chooses action based on a_dist values
            if np.random.rand() <= 0.01:
                a = np.random.randint(3)                                                    #move randomly every now and then
            a_v = action(a)
            s1 = np.add(s, a_v)                                                             #udpate the state based on the aciton chosen
            r = reward(s1)*1000/(1+j)                                                       #determind the reward of the state
            if j>=max_ep-1:                                                                 #if the agent has not hit a reflection in max_ep moves, it has failed miserably 
                print("Failure")
                total.append(max_ep)
                r=-10
            s1 = [s1[0]%360, s1[1]%360]                                                     #ensure that the state has not gone past 360 degrees (loops over)
            ep_history.append([s, a, r, s1])                                                #appends the state-reward-action-newstate list to the history
            s = s1                                                                          #update the state
            running_reward+=r  
            if j%10000 == 0:                                                                #prints attributes every 10,000 moves 
                print("a_dist: ", a_dist, "a: ", a, " L1: ", l1, s, i, r)
            if finished(s):                     
                """
                if a reflection is hit, the agent needs to attempt to learn why, and how to do so again
                """
                ep_history = np.array(ep_history)
                r = (max_ep - j)/10000
                np.append(ep_history[2], r)
                total.append(j)
                print();print();print();print();print(s, "Success, ", j);print();print();print();print();
                ep_history = np.array(ep_history)
                ep_history[:,2] = discounted_rewards(ep_history[:, 2])
                print(ep_history[2])
                feed_dict = {myAgent.reward_holder:ep_history[:,2], myAgent.action_holder:ep_history[:,1], myAgent.state_in:np.vstack(ep_history[:,0])}
                grads=sess.run(myAgent.gradients, feed_dict=feed_dict)
                for idx, grad in enumerate(grads):
                    gradBuffer[idx] += grad               

                if i%update_freqency == 0 and i!=0:
                    feed_dict = dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))        #updates the agents weights and therefor, outputs
                    _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                    for ix, grad in enumerate(gradBuffer):
                        gradBuffer[ix] = grad*0
                        
                total_reward.append(running_reward)
                print("a_dist: ", a_dist, "a: ", a, " L1: ", l1, s, i, r)
                break  
            
#----------------------------------------------------------- portion for finding the second reflection, not yet finished ---------------------------------------------------
            
#                if finished2(s):
#                    print("ROUND 2 FINISHED, A SECOND PEAK IS HIT")
#                    print(s);print();print();print()
#                    ep_history = np.array(ep_history)
#                    r = (max_ep - j)/10000
#                    print(ep_history[:,2])
#                    np.append(ep_history[2], r)
#                    print(ep_history[:,2])
#                    total.append(j)
#                    print();print();print();print();print(s, "Success, ", j);print();print();print();print();
##                    time.sleep(2)
#                    print(ep_history[:,2], r)
#                    ep_history = np.array(ep_history)
#                    ep_history[:,2] = discounted_rewards(ep_history[:, 2])
#                    print(ep_history[2])
#                    feed_dict = {myAgent.reward_holder:ep_history[:,2], myAgent.action_holder:ep_history[:,1], myAgent.state_in:np.vstack(ep_history[:,0])}
#                    grads=sess.run(myAgent.gradients, feed_dict=feed_dict)
#                    for idx, grad in enumerate(grads):
#                        gradBuffer[idx] += grad               
#
#                    if i%update_freqency == 0 and i!=0:
#                        feed_dict = dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
#                        _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
#                        for ix, grad in enumerate(gradBuffer):
#                            gradBuffer[ix] = grad*0
#                    s = [0,0]
#                    break  #out of for loop
                
## ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

                if i%100 == 0:
                    print(sess.run(myAgent.loss, feed_dict={myAgent.reward_holder:ep_history[:,2], myAgent.action_holder:ep_history[:,1], myAgent.state_in:np.vstack(ep_history[:,0])}), 'LOSS')
                    print(np.mean(total[-100:]), "a")
            if i%100==0 and i>0 and j==0:
                print(total, i, "k")
        i=i+1




















