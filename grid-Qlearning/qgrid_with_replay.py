import numpy as np
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import h5py
from collections import deque
import random

#startdefine grids

def randPair(s,e):
    return np.random.randint(s,e), np.random.randint(s,e)

#finds an array in the "depth" dimension of the grid
def findLoc(state, obj):
    for i in range(0,4):
        for j in range(0,4):
            if (state[i,j] == obj).all():
                return i,j

#Initialize stationary grid, all items are placed deterministically
def initGrid():
    state = np.zeros((4,4,4))#coordinates and state
    #place player
    state[0,1] = np.array([0,0,0,1])
    #place wall
    state[2,2] = np.array([0,0,1,0])
    #place pit
    state[1,1] = np.array([0,1,0,0])
    #place goal
    state[3,3] = np.array([1,0,0,0])

    return state

#Initialize player in random location, but keep wall, goal and pit stationary
def initGridPlayer():
    state = np.zeros((4,4,4))
    #place player
    state[randPair(0,4)] = np.array([0,0,0,1])
    #place wall
    state[2,2] = np.array([0,0,1,0])
    #place pit
    state[1,1] = np.array([0,1,0,0])
    #place goal
    state[1,2] = np.array([1,0,0,0])

    a = findLoc(state, np.array([0,0,0,1])) #find grid position of player (agent)
    w = findLoc(state, np.array([0,0,1,0])) #find wall
    g = findLoc(state, np.array([1,0,0,0])) #find goal
    p = findLoc(state, np.array([0,1,0,0])) #find pit
    if (not a or not w or not g or not p):
        print('Invalid grid. Rebuilding..')
        return initGridPlayer()

    return state

#Initialize grid so that goal, pit, wall, player are all randomly placed
def initGridRand():
    state = np.zeros((4,4,4))
    #place player
    state[randPair(0,4)] = np.array([0,0,0,1])
    #place wall
    state[randPair(0,4)] = np.array([0,0,1,0])
    #place pit
    state[randPair(0,4)] = np.array([0,1,0,0])
    #place goal
    state[randPair(0,4)] = np.array([1,0,0,0])

    a = findLoc(state, np.array([0,0,0,1]))
    w = findLoc(state, np.array([0,0,1,0]))
    g = findLoc(state, np.array([1,0,0,0]))
    p = findLoc(state, np.array([0,1,0,0]))
    #If any of the "objects" are superimposed, just call the function again to re-place
    if (not a or not w or not g or not p):
        print('Invalid grid. Rebuilding..')
        return initGridRand()

    return state

def makeMove(state, action):
    #need to locate player in grid
    #need to determine what object (if any) is in the new grid spot the player is moving to
    player_loc = findLoc(state, np.array([0,0,0,1]))
    #print player_loc
    wall = findLoc(state, np.array([0,0,1,0]))
    goal = findLoc(state, np.array([1,0,0,0]))
    pit = findLoc(state, np.array([0,1,0,0]))
    #print pit
    state = np.zeros((4,4,4))

    #up (row - 1)
    if action==0:
        new_loc = (player_loc[0] - 1, player_loc[1])
        if (new_loc != wall):
            if ((np.array(new_loc) <= (3,3)).all() and (np.array(new_loc) >= (0,0)).all()):
                state[new_loc][3] = 1
    #down (row + 1)
    elif action==1:
        new_loc = (player_loc[0] + 1, player_loc[1])
        if (new_loc != wall):
            if ((np.array(new_loc) <= (3,3)).all() and (np.array(new_loc) >= (0,0)).all()):
                state[new_loc][3] = 1
    #left (column - 1)
    elif action==2:
        new_loc = (player_loc[0], player_loc[1] - 1)
        if (new_loc != wall):
            if ((np.array(new_loc) <= (3,3)).all() and (np.array(new_loc) >= (0,0)).all()):
                state[new_loc][3] = 1
    #right (column + 1)
    elif action==3:
        new_loc = (player_loc[0], player_loc[1] + 1)
        if (new_loc != wall):
            if ((np.array(new_loc) <= (3,3)).all() and (np.array(new_loc) >= (0,0)).all()):
                state[new_loc][3] = 1

    new_player_loc = findLoc(state, np.array([0,0,0,1]))
    if (not new_player_loc):
        state[player_loc] = np.array([0,0,0,1])
    #re-place pit
    state[pit][1] = 1
    #re-place wall
    state[wall][2] = 1
    #re-place goal
    state[goal][0] = 1

    return state

def getLoc(state, level):#similar to findLoc
    for i in range(0,4):
        for j in range(0,4):
            if (state[i,j][level] == 1):
                return i,j

def getReward(state):
    player_loc = getLoc(state, 3)
    pit = getLoc(state, 1)
    goal = getLoc(state, 0)
    if (player_loc == pit):
        return -10
    elif (player_loc == goal):
        return 10
    else:
        return -1

def dispGrid(state):
    grid = np.chararray((4,4))
    player_loc = findLoc(state, np.array([0,0,0,1]))
    wall = findLoc(state, np.array([0,0,1,0]))
    goal = findLoc(state, np.array([1,0,0,0]))
    pit = findLoc(state, np.array([0,1,0,0]))
    for i in range(0,4):
        for j in range(0,4):
            grid[i,j] = 'O'
    if player_loc:
        grid[player_loc] = 'P' #player
    if wall:
        grid[wall] = 'W' #wall
    if goal:
        grid[goal] = '+' #goal
    if pit:
        grid[pit] = '-' #pit
    print grid
    return grid
    
#finish defining grids

def main():
    net = Sequential()
    net.add(Dense(164,input_shape = (64,)))
    net.add(Activation('relu'))
    net.add(Dense(150))
    net.add(Activation('relu'))
    net.add(Dense(4))
    net.add(Activation('linear'))
    try:
        net.load_weights("myweight.h5")
    except:
        print "weight load error"
    rms = RMSprop()
    net.compile(optimizer = rms, loss = 'mse')

    epoch = 100
    epsilon = 1
    gamma = 0.9
    batch_size = 40
    buffer = 80
    replay = deque()
    for i in xrange(epoch):
        state = initGrid()
        in_the_game = True  #status flag
        while (in_the_game):
            q = net.predict(state.reshape(1,64),batch_size = 1)
            if np.random.random() < epsilon:#explore or exploit
                action = np.random.randint(4)
            else:
                action = np.argmax(q)
            new = makeMove(state, action)
            reward = getReward(new)
            newq = net.predict(new.reshape(1,64),batch_size = 1)
            #save to the replay
            replay.append((state,action,reward,new))
           	if (len(replay) = buffer):
           		replay.popleft()

           	minibatch = random.sample(replay,batch_size)
           	x_train = []
           	y_train = []
           	for memory in minibatch:
	           	old_state, action, reward, new_state = memory
	           	old_qval = net.predict(old_state.reshape(1,64), batch_size=1)
                newQ = net.predict(new_state.reshape(1,64), batch_size=1)
                maxQ = np.max(newQ)
                y = old_qval.copy()
	            if reward == -1: #getting training data
	                y_real = reward + gamma * maxQ
	            else:
	                y_real = reward
	            y[0][action] = y_real
	            x_train.append(old_state.reshape(64))
	            y_train.append(y.reshape(4))
	         x_train = np.array(x_train)
	         y_train = np.array(y_train)   

            net.fit(x_train, y_train, batch_size = batch_size, nb_epoch = 1)
            state = new.copy()
            if (reward != -1):
                in_the_game = False #quit game
            if epsilon > 0.1:
                epsilon -= 1/epoch
        net.save_weights('myweight.h5')

    state = initGrid()
    print("Initial State:")
    print(dispGrid(state))
    status = 1
    i = 0
    #while game still in progress
    while(status == 1):
        qval = net.predict(state.reshape(1,64), batch_size=1)
        action = (np.argmax(qval)) #take action with highest Q-value
        print('Move #: %s; Taking action: %s' % (i, action))
        state = makeMove(state, action)
        dispGrid(state)
        reward = getReward(state)
        if reward != -1:
            status = 0
            print("Reward: %s" % (reward,))
        i += 1 #If we're taking more than 10 actions, just stop, we probably can't win this game
        if (i > 10):
            print("Game lost; too many moves.")
            break




if __name__ == "__main__":
   	main()
