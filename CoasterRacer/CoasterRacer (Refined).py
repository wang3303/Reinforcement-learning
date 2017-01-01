import gym
import universe # register Universe environments into Gym
import skimage as skimage
from skimage import transform, color, exposure,io
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt
import argparse
import random
import numpy as np
from collections import deque
import lutorpy as lua
lua.LuaRuntime(zero_based_index=False)
require("nn")
torch.setnumthreads(8)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1)

GAMMA = 0.9
GAME = 'flashgames.CoasterRacer2Bike-v0'
REPLAY_MEMORY = 1000
OBSERVE = 500
EPSILON_BEGIN = 0.5
EPSILON_END = 0.01
img_channels = 4 #We stack 4 frames
FRAME_PER_ACTION = 4
EXPLORE = 3000000
BATCH = 32 #size of minibatch

left = ('KeyEvent', 'ArrowLeft', True)
left_release = ('KeyEvent', 'ArrowLeft', False)
right =('KeyEvent', 'ArrowRight', True)
right_release =('KeyEvent', 'ArrowRight', False)
up = ('KeyEvent', 'ArrowUp', True)
up_release = ('KeyEvent', 'ArrowUp', False)
down = ('KeyEvent', 'ArrowDown', True)
down_release = ('KeyEvent', 'ArrowDown', False)
action_space = [[left,right,up,down],
			[left_release,right_release,up_release,down_release]]

def image_processing(image):
	image = skimage.color.rgb2gray(image)
	image = skimage.transform.resize(image,(80,80))
	image = skimage.exposure.rescale_intensity(image,out_range=(0,1))
	#image = image.reshape(1, image.shape[0], image.shape[1])
	return image

def network_building():
	model = nn.Sequential()
	model._add(nn.SpatialConvolution(img_channels,32,8,8,4,4))
	model._add(nn.ReLU())
	model._add(nn.SpatialConvolution(32,64,4,4,2,2))
	model._add(nn.ReLU())
	model._add(nn.SpatialConvolution(64,64,3,3,1,1))
	model._add(nn.ReLU())
	model._add(nn.View(64*6*6))
	model._add(nn.Linear(64*6*6,512))
	model._add(nn.ReLU())
	model._add(nn.Linear(512,2))

	model.criterion = nn.MSECriterion()
	return model


	

def train(model,args):
	#initialization
	env = gym.make(GAME) # any Universe environment ID here
	# If using docker-machine, replace "localhost" with your Docker IP
	env.configure(remotes='vnc://localhost:5900+15900')
	observation_n = env.reset()
	action_n = [[up] for ob in observation_n]  # press the up key
	observation_n, reward_n, done_n, info = env.step(action_n)
	env.render()
	memory = deque()

	while observation_n[0] == None: # wait for game to begin
		observation_n, reward_n, done_n, info = env.step(action_n)
		env.render()
		
	action_n = [[up] for ob in observation_n]  # press the up key
	observation_n, reward_n, done_n, info = env.step(action_n)
	env.render()

	frame1 = observation_n[0]['vision']
	frame1 = image_processing(frame1)
	frame1 = np.stack((frame1,frame1,frame1,frame1),axis = 0)#stack four frames
	frame1 = frame1.reshape(1,frame1.shape[0],frame1.shape[1],frame1.shape[2])

	if args['mode'] == 'Run': # evaluation mode:
		OBSERVE = 999999999 
		epsilon = EPSILON_END
	else:
		OBSERVE = 33
		epsilon = EPSILON_BEGIN
	#done_n = False #initial state

	t = 0 #counter
	
	for episode in xrange(1000):
		print "episode %d" % episode
		while True:
			Q = 0
			loss = 0
			reward_n = 0
			if t % FRAME_PER_ACTION ==0:
				if np.random.random() < epsilon:
					print "RANDOM ACTION"
					action_index = np.random.randint(0,2)
					
					#action_n = [[action_space[0][action_index]]]
				else:
					q = model._forward(torch.fromNumpyArray(frame1)._float()) #input a stack of 4 images
					maxQ = np.argmax(q.asNumpyArray())
					action_index = maxQ
					#action_n = [[action_space[0][action_index]]]
					#action_n = [[('KeyEvent', 'ArrowUp', True)] for _ in observation_n]
					#observation_n, reward_n, done_n, info = env.step(action_n)
			#decaying epsilon
			if epsilon > EPSILON_END and t > OBSERVE:
				epsilon -= (EPSILON_BEGIN - EPSILON_END) / EXPLORE
			#PRESS THE KEY AND RELEASE
			observation_n, reward_n, done_n, info = env.step([[action_space[0][action_index]]])
			env.render()
			observation_n, reward_n, done_n, info = env.step([[action_space[1][action_index]]])
			env.render()

			if observation_n[0] == None:
				break

			frame2 = observation_n[0]['vision']
			frame2 = image_processing(frame2)
			frame2 = frame2.reshape(1,1,frame2.shape[0],frame2.shape[1])
			frame2 = np.append(frame2,frame1[:,:3,:,:],axis = 1)
			memory.append((frame1,action_index,reward_n[0],frame2,done_n[0]))
			if len(memory) > REPLAY_MEMORY:
				memory.popleft()

			if t > OBSERVE:
				minibatch = random.sample(memory,BATCH)
				x_train = np.zeros((BATCH,frame1.shape[1],frame1.shape[2],frame1.shape[3]))
				y_train = np.zeros((BATCH,2))
				for i in xrange(0, BATCH):
					state = minibatch[i][0]
					action = minibatch[i][1]
					reward = minibatch[i][2]
					state1 = minibatch[i][3]
					done = minibatch[i][4]

					x_train[i:i+1] = state
					y_train[i] = model._forward( torch.fromNumpyArray(state)._float() ).asNumpyArray()
					Q = model._forward( torch.fromNumpyArray(state1)._float() ).asNumpyArray()
					if done:
						update = reward
					else:
						update = reward + GAMMA * np.max(Q)
					y_train[action] = update
				
				#training
				ii = torch.fromNumpyArray(x_train)._float()
				oo = model._forward(ii)
				tt = torch.fromNumpyArray(y_train)._float()

				model._zeroGradParameters()
				loss = loss + model.criterion._forward(oo, tt)
				dE_dy = model.criterion._backward(oo, tt)
				model._backward(ii, dE_dy)
				model._updateParameters(0.1)

			#post_process
			frame2 = frame1
			t += 1

			state = ""
			if t <= OBSERVE:
				state = "observe"
			elif t > OBSERVE and t <= OBSERVE + EXPLORE:
				state = "explore"
			else:
				state = "train"
			
			print("TIMESTEP", t,\
				"/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", reward_n, \
				"/ Q_MAX " , np.max(Q), "/ Loss ", loss)
			

def play(args):
	print "building model"
	net = network_building()
	train(net,args)

def main():
	parser = argparse.ArgumentParser(description='Description of your program')
	parser.add_argument('-m','--mode', help='Train / Run', required=True)
	args = vars(parser.parse_args())
	play(args)

if __name__ == "__main__":
	main()