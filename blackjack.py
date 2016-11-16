import numpy as np 
import random
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#define environment
#a state of hand is tuple(sum, whether there is a useable ace)
def randomcard():
	card = np.random.randint(1,14)
	if card > 10:
		card = 10
	return card

#you have to have a ace and it doesn't bust
def useable_ace(hand):
	val, ace = hand
	return ((ace) and (val + 10 <= 21)) 

#get the current max value of cards
def total_value(hand):
	val, ace = hand
	if useable_ace(hand):
		return (val + 10)
	else: 
		return val

def add_card(hand, card):
	val, ace = hand
	if card == 1:
		ace = True
	return (val + card, ace)

#get the dealer's score
def dealer(hand):
	while(total_value(hand) < 17):
		hand = add_card(hand, randomcard())
	return hand #(this is a tuple)

#distribute cards to dealers and players
def init():
	status = 1#1=in progress; 2=player won; 3=draw; 4 = dealer won/player loses
	player_hand = (0,False)
	dealer_hand = (0,False)
	player_hand = add_card(player_hand,randomcard())
	player_hand = add_card(player_hand,randomcard())
	dealer_hand = add_card(dealer_hand,randomcard())
	if total_value(player_hand) == 21:#player got 21
		if total_value(dealer_hand) != 21:
			status = 2
		else:
			status = 3
	return (player_hand,dealer_hand,status)

def play(state,hit):#hit is a flag
	player_hand = state[0]
	dealer_hand = state[1]
	status = state[2]
	p_total = total_value(player_hand)
	dealer_tot = total_value(dealer_hand)
	if (not hit):
		#print "not hit"
		dealer_hand = dealer(dealer_hand)
		p_total = total_value(player_hand)
                dealer_tot = total_value(dealer_hand)
                status = 1
                if (dealer_tot > 21):
                    status = 2 #player wins
                elif (dealer_tot == p_total):
                    status = 3 #draw
                elif (dealer_tot < p_total):
                    status = 2 #player wins
                elif (dealer_tot > p_total):
                    status = 4
	else:
		#print "hit"
		player_hand = add_card(player_hand, randomcard())
                d_hand = dealer(dealer_hand)
                p_total = total_value(player_hand)
                status = 1
                if (p_total == 21):
                    if (total_value(d_hand) == 21):
                        status = 3 #draw
                    else:
                        status = 2 #player wins!
                elif (p_total > 21):
                    status = 4 #player loses
                elif (p_total < 21):
                        status = 1
	return (player_hand,dealer_hand,status)

def calcreward(status):
	return 3-status

def getRLstate(state):#that is what the player knows
	player_hand, dealer_hand, status = state
	player_val, player_ace = player_hand
	return (player_val, player_ace, dealer_hand[0])

def q(state,av_table):
	return np.array([av_table[(state,0)],av_table[(state,1)]])

def update(av_table,av_count,returns):
	for i in returns:
		av_table[i] = av_table[i]+(1.0/av_count[i])*(returns[i]-av_table[i])
	return av_table

def main():
	epochs = 5000000
	epsilon = 0.1 #exploitatio & exploration
	#(player's value, usable ace, dealer's card)
	state_space = []
	for card in range(1,11):
		for val in range(11,22):
			state_space.append((val,False,card))
			state_space.append((val,True,card))
	#state-action table
	av_table = {}
	for state in state_space:
		av_table[(state,0)] = 0.0
		av_table[(state,1)] = 0.0
	#count the time this stateaction is met FOR UPDATING
	av_count = {}
	for stateaction in av_table:
		av_count[stateaction] = 0

	#start playing
	for i in range(epochs):
		state = init()
		player_hand, dealer_hand, status = state
		while player_hand[0] < 11:
			player_hand = add_card(player_hand,randomcard())
			state = (player_hand, dealer_hand, status)
		rl_state = getRLstate(state) 

		returns = {}#to hold current epoch (state,action,reward)
		while(state[2] == 1):
			act_prob = q(rl_state,av_table)
			if np.random.rand() < epsilon:#choose action randomly
				action = np.random.randint(0,2)
			else:#choose action with greater value
				action = np.argmax(act_prob)
			av_count[(rl_state,action)] += 1
			returns[(rl_state,action)] = 0#add a-v pair to returns list, default value to 0
			state = play(state,action) #step further
			rl_state = getRLstate(state)
		for i in returns:
			returns[i] = calcreward(state[2])
		av_table = update(av_table,av_count,returns)
	print "finish %d rounds of blackjack" % epochs

	#ploting (no useable Aces are present
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d', )
	ax.set_xlabel('Dealer card')
	ax.set_ylabel('Player sum')
	ax.set_zlabel('State-Value')

	x,y,z = [],[],[]
	for key in state_space:
		if (not key[1]):
			x.append(key[2])
			y.append(key[0])
			z.append(np.max(np.array([av_table[(key,0)],av_table[(key,1)]])))
	ax.plot_trisurf(x,y,z, linewidth=.02, cmap=cm.jet)
	plt.show()

if __name__ == "__main__":
	main()

