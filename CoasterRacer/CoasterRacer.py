import gym
import universe # register Universe environments into Gym
import skimage as skimage
from skimage import transform, color, exposure,io
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import matplotlib.pyplot as plt
import deque
'''
import lutorpy as lua
lua.LuaRuntime(zero_based_index=False)
require("nn")
torch.setnumthreads(8)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(1)
'''

GAME = 'flashgames.CoasterRacer2Bike-v0'
REPLAY_MEMORY = 1000
OBSERVE = 1000
EPSILON_BEGIN = 
EPSILON_END = 

def image_processing(image):
	image = skimage.color.rgb2gray(image)
    image = skimage.transform.resize(image,(80,80))
    image = skimage.exposure.rescale_intensity(image,out_range=(0,1))
    image = image.reshape(1, 1, image.shape[0], image.shape[1])
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
    model._add(nn.Linear(512,4))

    model.criterion = nn.MSECriterion()
    return model

def train(net,args):
	#initialization
	env = gym.make(GAME) # any Universe environment ID here
	# If using docker-machine, replace "localhost" with your Docker IP
	env.configure(remotes='vnc://localhost:5900+15900')
	observation_n = env.reset()
	memory = deque()

	done_n = False #initial state
	while True:
		# agent which presses the Up arrow 60 times per second
		action_n = [[('KeyEvent', 'ArrowUp', True)] for _ in observation_n]
		observation_n, reward_n, done_n, info = env.step(action_n)
		

		if done_n:
			pass
		else:
		
		env.render()



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
	play()
