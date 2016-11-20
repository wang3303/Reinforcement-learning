# Reinforcement-learning
Getting started from basic reinforcement learning
# Year 2016

* 11/19-11/20
[Train on torcs racing game with sensor input](https://github.com/wang3303/Reinforcement-learning/tree/master/torcs) (preliminary, this program can be run at present)

further problem faced: 

	1. the response time should be below 10ms. my program has a lag. There might be some room for simplifyinh.

	2. The game is rather different from [outrun](https://lopespm.github.io/machine_learning/2016/10/06/deep-reinforcement-learning-racing-game.html) in that the action dimension is infinite.

	Solved: Implement actor-critic network structure

	3. Unstable performance

	Clue: build target network do reduce the updating step.

* 11/18

[Q-learning exercise(with and without replay)](https://github.com/wang3303/Reinforcement-learning/tree/master/grid-Qlearning)

* 11/17

autoencoder for mnist using keras

![autoencoder](https://github.com/wang3303/Reinforcement-learning/blob/master/fully_connected_autoencoder_mnist.png)

OpenAI [gym](https://github.com/wang3303/Reinforcement-learning/edit/master/gym_basics.py)

Succeed in finding problems. torcs environment is correctly set.

![environment setting](https://github.com/wang3303/Reinforcement-learning/blob/master/success)

* 11/15

work on blackjack problem using [Monte Carlo & Tabular Methods](http://outlace.com/Reinforcement-Learning-Part-2/)

![result of no usable aces](https://github.com/wang3303/Reinforcement-learning/blob/master/blackjack.png)

* 11/14

program simple n-bandit arm machine with python


 [reference](http://outlace.com/Reinforcement-Learning-Part-1/)
 
 * 11/11-11/13

Learn keras & matplotlib

* before 11/11

Try to set environment of gym_torcs (~~problem faced plib v1.8.3 configured not properly and that is the only version works~~)

Read [code](https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html) of DDPG of TORCS game

Review python and numpy

Install dependences and toolboxes

Learn lutorpy and argparse

Go through [Flappybird](https://yanpanlau.github.io/2016/07/10/FlappyBird-Keras.html)










