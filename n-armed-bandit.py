import matplotlib.pyplot as plt
import numpy as np
import random

n = 10
arms = np.random.rand(n)
eps = 0.1

#define environment
def reward(p):
	r = 0
	for i in range(n):
		if np.random.rand() < p:
			r += 1
	return r

#memory array
av = np.array([np.random.randint(0,n),0]).reshape(1,2)

#get best action
def get_best(memory):
	action = 0
	action_value = 0
	for u in range(10):
		mean = np.mean(memory[np.where(memory[:,0] == u)][:,1])#special technique
		if mean > action_value:
			action_value = mean
			action = u
	return action

def main():
	global av
	print arms
	plt.xlabel("plays")
	plt.ylabel("average reward")
	for i in range(500):
		if np.random.rand() > eps: #greedy
			action = get_best(av)
			add = np.array([[action,reward(arms[action])]])
			av = np.concatenate((av,add),axis = 0)
		else:
			action = np.random.randint(0,n) #random
			add = np.array([[action,reward(arms[action])]])
			av = np.concatenate((av,add),axis = 0)
		runningmean = np.mean(av[:,1])
		plt.scatter(i,runningmean)
		print action == np.argmax(arms)
	#print "THe number is %d" % np.argmax(arms)
	plt.show()

if __name__ == "__main__":
	main()