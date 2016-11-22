import os
os.environ['KERAS_BACKEND']='tensorflow'

import numpy as np 
from gym_torcs import TorcsEnv
import random
from collections import deque
from keras.models import Model, model_from_json, Sequential
from keras.layers import Dense, Dropout, Activation,Input,merge
from keras.optimizers import Adam
import h5py
from keras.initializations import normal
import tensorflow as tf 
from keras.engine.training import collect_trainable_weights

#Tensorflow GPU optimization
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)

def build_agent_network(state_dim, action_dim): #map state to action
	#function for initialization
	my_init = lambda shape, name = None:normal(shape, scale = 1e-4, name = name)  
	s0 = Input(shape=(state_dim,))
	s1 = Dense(300,activation = 'relu')(s0)
	s2 = Dense(600,activation = 'relu')(s1)
	steer = Dense(1,activation = 'tanh',init = my_init)(s2)#(-1,1)
	acc = Dense(1,activation = 'sigmoid',init = my_init)(s2)#(0,1)
	brake = Dense(1,activation = 'sigmoid', init = my_init)(s2)#(0,1)
	action = merge([steer,acc,brake],mode = 'concat')
	model = Model(input = s0, output = action)
	#adam = Adam(lr=1e-4)
    #model.compile(loss='mse', optimizer=adam)
	return model, model.trainable_weights,s0

def build_critic_network(state_size,action_dim):
    S = Input(shape=[state_size])  
    A = Input(shape=[action_dim],name='action2')   
    w1 = Dense(300, activation='relu')(S)
    a1 = Dense(600, activation='linear')(A) 
    h1 = Dense(600, activation='linear')(w1)
    h2 = merge([h1,a1],mode='sum')    
    h3 = Dense(600, activation='relu')(h2)
    V = Dense(action_dim,activation='linear')(h3)   
    model = Model(input=[S,A],output=V)
    adam = Adam(lr=1e-4)
    model.compile(loss='mse', optimizer=adam)	
    return model,A,S

def noise(x,mu, theta, sigma):
	return theta * (mu - x) + sigma * np.random.randn(1)

def playgame(train = 1):
	buffer_size = 900
	memory = deque()
	gamma = 0.98
	batch_size = 40
	action_dim = 3	#Three actions:steer/accelerate/brake
	state_dim = 29 #29 sensors
	step = 0
	np.random.seed(1) #for replay

	vision = False #not accepting real-time picture

	epsilon = 1 #possibility of choosing randomly(gradually decreasing)
	epoch = 10000

	#build network
	net,agent_weights,agent_state = build_agent_network(state_dim,action_dim)
	critic, critic_action, critic_state = build_critic_network(state_dim,action_dim)
	action_gradient = tf.placeholder(tf.float32,[None, action_dim])
	params_grad = tf.gradients(net.output, agent_weights, -action_gradient)
	grads = zip(params_grad, agent_weights)
	optimze = tf.train.AdamOptimizer(1e-4).apply_gradients(grads)
	action_grads = tf.gradients(critic.output, critic_action)
	sess.run(tf.initialize_all_variables())

	
	try:
		net.load_weights("agent_weight.h5")
		critic.load_weights("critic_weight.h5")
		print "loading weights..."
	except:
		print "fail loading"

	#generate environment
	environment = TorcsEnv(vision = vision, throttle = True, gear_change = False)

	#start playingdd
	for i in xrange(epoch):
		print "epoch %d:" % i

		if np.mod(i,3):
			ob = environment.reset(relaunch = True) #avoid memory leak
		else:										#relaunch every 3 times
			ob = environment.reset()

		state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

		epoch_reward = 0
		for j in xrange(100000):#max step in a epoch is set to 100000
			if epsilon > 0.001:#decreasing randomness of action selection
				epsilon -= 1e-4
			action = np.zeros((1,3))
			action = net.predict(state.reshape(1,-1))
			action[0][0] += noise(action[0][0],0,0.6,0.3)*epsilon
			action[0][1] += noise(action[0][1],0.5,1,0.1)*epsilon
			action[0][2] += noise(action[0][1],-0.1,1,0.05)*epsilon

			ob, reward, done, info = environment.step(action.reshape(action_dim)) #obsercation, re
			new_state = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
			
			memory.append((state,action,reward,new_state,done))
			while len(memory) >= buffer_size:
				memory.popleft()
				
			print "length of memory"
			print len(memory)
			#updating
			minibatch_size = np.min((batch_size,len(memory)))
			minibatch = random.sample(memory,minibatch_size)
			x_state = np.asarray([e[0] for e in minibatch])
			x_action = np.asarray([e[1][0] for e in minibatch])
			r = np.asarray([e[2] for e in minibatch])
			batch_new_state = np.asarray([e[3] for e in minibatch])
			do = np.asarray([e[4] for e in minibatch])
			y_train = np.asarray([e[1][0] for e in minibatch])
			qval = critic.predict([batch_new_state.reshape(minibatch_size,-1),net.predict(batch_new_state.reshape(minibatch_size,-1))])
			for k in xrange(minibatch_size):
				if do[k]:
					y_train[k] = r[k]
				else:
					y_train[k] = r[k] + gamma*qval[k]
			'''
			#alternative method for constructing traning data for critic network
			x_state = []
			x_action = []
			y_train = [] 
			
			for replay in minibatch:
				batch_state,action,r,batch_new_state,do = replay
				fut = critic.predict([batch_new_state.reshape(1,-1),net.predict(batch_new_state.reshape(1,-1))])
				if do:
					qval = r * np.ones(np.shape(fut))
				else:
					qval = r + fut*gamma
				x_action.append(action.reshape(action_dim))
				x_state.append(batch_new_state.reshape(state_dim))
				#print np.shape(qval)
				y_train.append(qval.reshape(action_dim))

			x_state = np.array(x_state)
			x_action = np.array(x_action)
			y_train = np.array(y_train)
			'''

			if train:
				critic.fit([x_state,x_action],y_train,nb_epoch = 1, batch_size = minibatch_size)
				#critic.train_on_batch([x_state,x_action],y_train)
				res = net.predict(x_state)
				grad = sess.run(action_grads, feed_dict = {
				critic_state:x_state, critic_action:res
				})
				sess.run(optimze, feed_dict={agent_state:x_state,action_gradient : grad[0]})
			state = new_state
			print("Episode", i, "Step", step, "Reward", reward)
			step+=1
			if done:
				break
		if train:
			print "save weights..."
			net.save_weights("agent_weight.h5",overwrite = True)
			critic.save_weights("critic_weight.h5",overwrite = True)
	environment.end()
	if train:
		print "finish training"
		
def main():
	playgame()


if __name__ == "__main__":
	main()
