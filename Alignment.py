#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 16:41:53 2017

@author: Rohit Mandavia
"""

#import tensorflow as tf
import numpy as np
#import ubmatrix
import tensorflow as tf
import tensorflow.contrib.slim as slim
from ubmatrix import star, calcB, calcIdealAngles, calcU, calcUB
from arbitrary_rotation import rotation_a, initial, equivalence
import math
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

gam = 0.99

class agent():
    def __init__(self, lr, s_size, a_size, h_size):
        #take state and produce an action
        self.state_in = tf.placeholder(shape = [None, s_size], dtype=tf.float32, name = "STATE_IN")
        print(self.state_in, "STATE_IN")
        hidden = slim.fully_connected(self.state_in, h_size, biases_initializer = None,activation_fn = tf.nn.relu)
        print(hidden, "HIDDEN")
#        self.output = slim.fully_connected(hidden, a_size, activation_fn=tf.nn.softmax, biases_initializer=None)
        self.output = slim.fully_connected(hidden, a_size, activation_fn=tf.nn.softmax, biases_initializer=None)

        print(self.output, "OUTPUT")
        # determines which action is the best to take
        
        self.chosen_action = tf.argmax(self.output, 1, name = "CHOSEN_ACTION")  #selects the action that has the higest probabiltiy
        print(self.chosen_action, "CHOSEN ACTION")
        tf.summary.scalar("state_in", self.state_in)
        tf.summary.histogram("state_in_histogram", self.state_in)
        tf.summary.histogram("chosen_action", self.chosen_action)
        
        self.cost_holder = tf.placeholder(shape=[None], dtype=tf.float32, name = "COST_HOLDER")
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32, name = "REWARD_HOLDER")
        print(self.reward_holder, "REWARD HOLDER")
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32, name= "ACTION_HOLDER")
        print(self.action_holder, "ACTION HOLDER")


        self.o2 = tf.reshape(self.output, [-1])


        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        print(self.indexes, "INDEXES")
        self.responsible_outputs= tf.gather(tf.reshape(self.output, [-1]), self.indexes)  #the probabilities that were chosen
        print(self.responsible_outputs, "RESPONSIBLE OUTPUTS") 
        self.check1 = tf.log(self.responsible_outputs)*self.cost_holder
        self.check2 = tf.log(self.responsible_outputs)
        self.loss = tf.reduce_mean(tf.log(self.responsible_outputs)*self.cost_holder)    # the mean of the discounted reward times ln(responsible outputs)
        print(self.loss, "LOSS")
        tvars = tf.trainable_variables()                # The hidden net and the output layer
        print(tvars, "TVARS")
        self.gradient_holders = []
        for idx, var in enumerate(tvars):               #idx = index, var = layer
            print(idx, "IDX", var, "VAR")
            placeholder = tf.placeholder(tf.float32, name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        print(self.gradient_holders, 'GRADIENT_HOLDER')
        self.gradients = tf.gradients(self.loss, tvars)  #what does this even mean? Derivative of the loss with respect to the tvars???
        #partial derivative of the loss with respect to the tvars
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)  #the method of optimizing we use
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))  # updating by applying gradients using the layers (Tvars)
        # this is where the weights are changed i believe 

        
        
tf.reset_default_graph()
merged = tf.summary.merge_all()
myAgent = agent(lr = 0.1, s_size = 2, a_size = 6, h_size = 8)  # state size, action size, hidden


def action(a):
    return [[0,0],[1,0],[0,1],[1,1],[-1,0],[-1,-1]][a]  # might want to include [0,0]

def finished(possibilities, x_c, p_c, o_c=0, eps=0.5):
    """
    given the possibile combinations of motor positions for a reflection, this method determines if the agent has found a peak.
    """
    
    if 100<x_c<200 and 100 < p_c < 150:
        return True
    for i in range(len(possibilities)):
        if possibilities[i][0]+eps >= o_c >= possibilities[i][0]-eps and possibilities[i][1]+eps >=x_c>= possibilities[i][1]-eps and possibilities[i][2]+eps >= p_c >= possibilities[i][2]-eps: 
            return True
    return False


def discounted_cost(c):
    discounted_c = np.zeros_like(c)
    running_add = 0
    for t in reversed(range(0, c.size)):
        running_add = running_add*gam + c[t]
        discounted_c[t] = running_add
    return discounted_c

def create_data(z):
    mag_z = np.sqrt(z[0]**2 + z[1]**2 + z[2]**2)
    u1 = [z[0]/mag_z, z[1]/mag_z, z[2]/mag_z]
#    mag_y = np.sqrt(6)
#    u2_i = [1/mag_y, 1/mag_y, -2/mag_y]
    u2_i = [0,1,0] 
    
    for i in range(360):        
        u2 = rotation_a(u1, u2_i, i)
        u3 = np.cross(u1, u2)
        print(u1, u2, u3)
        if i%30 == 0:
            soa = np.array([np.append([0,0,0], u1), np.append([0,0,0], u2), np.append([0,0,0], u3)])

            X, Y, Z, U, V, W = zip(*soa)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.quiver(X, Y, Z, U, V, W)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_zlim([0, 1])
            plt.show()
            plt.clf()
        
        #write these to a file ^ for future reference        plt.o
    

def get_inital_hkl_data():
    data = open('data/test.txt', 'r')
    MAT, HKLs, Structure_factors, theta, two_theta = [],[],[],[],[]
    for line in data:
        words = line.split(",")
        MAT.append(words)
    for i in range(len(MAT)-1):
        Structure_factors.append(float(MAT[i][5].strip()))
        theta.append(float(MAT[i][3].strip()))
        two_theta.append(float(MAT[i][4].strip()))
        HKLs.append([MAT[i][0], MAT[i][1], MAT[i][2]])
    return HKLs, Structure_factors, theta, two_theta 

if __name__ == "__main__":
    # define latice parameters
    a, b, c, alpha, beta, gamma = 3.9, 3.9, 3.9, 90, 90, 90
    a_S, b_S, c_S, alpha_S, beta_S, gamma_S = star(a, b, c, alpha, beta, gamma)
    stars = [a_S, b_S, c_S, alpha_S, beta_S, gamma_S]
    stars_dict = dict(list(zip(('astar','bstar','cstar','alphastar','betastar','gammastar'),stars)))
    HKLs, structure_factor, theta, two_theta = get_inital_hkl_data()
    recip = (a_S, b_S, c_S, alpha_S, beta_S, gamma_S, c, alpha)
    B_matrix = calcB(*recip)

    print();print();print("----------------------------------------------------------------------------");print();print()
    
    sf = np.array(structure_factor)
    
    motor_angles = []
    equivs = equivalence([1,0,0])    
    i = initial([0,0,1], [1,0,0], 17.59, B_matrix)
    U = calcU(*i)
    UB = calcUB(*i)
    
    # These motor angles are all the combinations of motor positions that will lead to a reflection. 
    # This means that if the motor angles are at these values, then there will be a peak there for the specific 2_Theta value
    for j in range(len(equivs)):
        angles = np.array(calcIdealAngles(equivs[j], UB, B_matrix, 2.359, stars_dict)[2:])
        for i in range(len(angles)):
            angles[i] = (angles[i])%360
        motor_angles.append(angles)
    print(finished(motor_angles, 0, 0, 18))
    
    
   
    
    
    
total_episodes = 50
update_freqency = 5

init = tf.global_variables_initializer()
    
    
        
with tf.Session() as sess:
    sess.run(init)
    i = 0
    total_cost = []
    total_reward = []
    
    gradBuffer = sess.run(tf.trainable_variables())
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad*0                 # why not just 0, maybe because '-0' vs '0'
    while i < total_episodes:
        omega, chi, phi = 0,0,0                 # initial motor positions
        s = [chi, phi]                   # the state
        running_reward, running_cost = 0,0
        ep_history = []
        frame=0
  

        # this entire for loop is one game play. 
        while not (finished(motor_angles, omega, *s)):
            frame+=1
            s=[s[0]%360, s[1]%360]
#            print(s, motor_angles)
            r = -1
            cost = 1
            a_dist = sess.run(myAgent.output, feed_dict={myAgent.state_in:[s]})   #feed in the state to determe the output, runs through the hidden layer and the output layer
            print(a_dist[0])
            a = np.random.choice(a_dist[0], p=a_dist[0])        # picks the left or right number based on the size of the number: [.344, .666] -> 34.4% chance a is .344, and 66.6% chance a is .666
#            a = np.argmax(a_dist==a)                            # the position of the true: [False, True] --> 1 (move right)
            l1 = []
            for i in range(len(a_dist[0])):
                if (a_dist[0][i] == a):
                    l1.append(i)
            a = np.random.choice(l1)
#            a = np.random.choice(6)
            a_v = action(a)
            s1 = np.add(s,a_v)
            if (frame%20 == 0):
                print(s, a)
#            print(a)
#            print(s1, a_v)
            if(frame%1000 == 0):
                print(a_dist[0])
                A_ep_history = np.array(ep_history)
                feed_dict = {myAgent.cost_holder:A_ep_history[:,2], myAgent.action_holder:A_ep_history[:,1], myAgent.state_in:np.vstack(A_ep_history[:,0])}
#                print(sess.run(myAgent.chosen_action, feed_dict=feed_dict), "CHOSEN ACTION", len(sess.run(myAgent.chosen_action, feed_dict=feed_dict)))
#                print(sess.run(myAgent.action_holder, feed_dict=feed_dict), "ACTION HOLDER", len(sess.run(myAgent.action_holder, feed_dict=feed_dict)))
#                print(sess.run(myAgent.indexes, feed_dict=feed_dict), "INDEXES", len(sess.run(myAgent.indexes, feed_dict=feed_dict)))
#                print(sess.run(myAgent.responsible_outputs, feed_dict=feed_dict), "RESPONSIBLE OUTPUTS", len(sess.run(myAgent.responsible_outputs, feed_dict=feed_dict)))
#                print(sess.run(myAgent.loss, feed_dict=feed_dict), "LOSS")

            if (finished(motor_angles, chi, phi)):
                time.sleep(3)
                r=1
            ep_history.append([s, a, cost, s1])        # appends the state-reward-action-newstate list to the history
            s = s1                                  # update the state
            running_reward+=r                       # update the reward
            running_cost += 1
            if finished(motor_angles, *s):
                ep_history = np.array(ep_history)
                ep_history[:,2] = discounted_cost(ep_history[:, 2])
#                print();print(); print(ep_history[:,2]); print(ep_history[:,1]); print(np.vstack(ep_history[:,0]));
                feed_dict = {myAgent.cost_holder:ep_history[:,2], myAgent.action_holder:ep_history[:,1], myAgent.state_in:np.vstack(ep_history[:,0])}
                print(feed_dict)
                grads=sess.run(myAgent.gradients, feed_dict=feed_dict)
                print(grads)
                time.sleep(3)
                print();print();print(running_cost);print();print()
#                print(len(grads), len(grads[0]), len(grads[0][0]), len(grads[0][0]), len(grads[0][1]), len(grads[1]), len(grads[1][0]), len(grads[1][1]), len(grads[2][2]))
                for idx, grad in enumerate(grads):
                    gradBuffer[idx] += grad
#                    print(idx,grad)
#                print(gradBuffer, len(gradBuffer))
#                time.sleep(5)
#                if i%update_freqency == 0 and i!=0:
                feed_dict = dictionary = dict(zip(myAgent.gradient_holders, gradBuffer))
                _ = sess.run(myAgent.update_batch, feed_dict=feed_dict)
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad*0
                print("-------------------------------UPDATED---------------------------------------")
                time.sleep(1)
                        
                total_reward.append(running_reward)
#                total_length.append(j)
                break  #out of while loop?
        if i % 10 == 0:
            print(sess.run(myAgent.loss, feed_dict={myAgent.reward_holder:ep_history[:,2], myAgent.action_holder:ep_history[:,1], myAgent.state_in:np.vstack(ep_history[:,0])}), 'LOSS')
            print(np.mean(total_reward[-10:]), "a")
            time.sleep(1)
#            print(sess.run(myAgent.update_batch, feed_dict=feed_dict))
        if i%1000==0:
            print(total_reward)
#            print(total_length)
        print(a)
        time.sleep(0.01)
        i+=1
    