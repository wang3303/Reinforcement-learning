# Reinforcement-learning
Getting started from basic reinforcement learning
# Year 2016
* 12/22
Details:
```
observation_n, reward_n, done_n, info = env.step(action_n)
# observation_n, reward_n, done_n,info are all lists
# observation_n = [{'text':[],'vision':[,dtype = uint8]}]
# reward_n = [reward] 
# done_n = [false] 
# Each environment's info message contains useful diagnostic information, including latency data, client and remote timings, VNC update counts, and reward message counts.
```
```
KEYMAP = {
    'bsp': KEY_BackSpace,
    'tab': KEY_Tab,
    'return': KEY_Return,
    'enter': KEY_Return,
    'esc': KEY_Escape,
    'ins': KEY_Insert,
    'delete': KEY_Delete,
    'del': KEY_Delete,
    'home': KEY_Home,
    'end': KEY_End,
    'pgup': KEY_PageUp,
    'pgdn': KEY_PageDown,
    'ArrowLeft': KEY_Left,
    'left': KEY_Left,
    'ArrowUp': KEY_Up,
    'up': KEY_Up,
    'ArrowRight': KEY_Right,
    'right': KEY_Right,
    'ArrowDown': KEY_Down,
    'down': KEY_Down,

    'slash': KEY_BackSlash,
    'bslash': KEY_BackSlash,
    'fslash': KEY_ForwardSlash,
    'spacebar': KEY_SpaceBar,
    'space': KEY_SpaceBar,
    'sb': KEY_SpaceBar,

    'f1': KEY_F1,
    'f2': KEY_F2,
    'f3': KEY_F3,
    'f4': KEY_F4,
    'f5': KEY_F5,
    'f6': KEY_F6,
    'f7': KEY_F7,
    'f8': KEY_F8,
    'f9': KEY_F9,
    'f10': KEY_F10,
    'f11': KEY_F11,
    'f12': KEY_F12,
    'f13': KEY_F13,
    'f14': KEY_F14,
    'f15': KEY_F15,
    'f16': KEY_F16,
    'f17': KEY_F17,
    'f18': KEY_F18,
    'f19': KEY_F19,
    'f20': KEY_F20,

    'lshift': KEY_ShiftLeft,
    'shift': KEY_ShiftLeft,
    'rshift': KEY_ShiftRight,
    'lctrl': KEY_ControlLeft,
    'ctrl': KEY_ControlLeft,
    'rctrl': KEY_ControlRight,
    'lmeta': KEY_MetaLeft,
    'meta': KEY_MetaLeft,
    'rmeta': KEY_MetaRight,
    'lalt': KEY_AltLeft,
    'alt': KEY_AltLeft,
    'ralt': KEY_AltRight,
    'scrlk': KEY_Scroll_Lock,
    'sysrq': KEY_Sys_Req,
    'numlk': KEY_Num_Lock,
    'caplk': KEY_Caps_Lock,
    'pause': KEY_Pause,
    'lsuper': KEY_Super_L,
    'super': KEY_Super_L,
    'rsuper': KEY_Super_R,
    'lhyper': KEY_Hyper_L,
    'hyper': KEY_Hyper_L,
    'rhyper': KEY_Hyper_R,

    'kp0': KEY_KP_0,
    'kp1': KEY_KP_1,
    'kp2': KEY_KP_2,
    'kp3': KEY_KP_3,
    'kp4': KEY_KP_4,
    'kp5': KEY_KP_5,
    'kp6': KEY_KP_6,
    'kp7': KEY_KP_7,
    'kp8': KEY_KP_8,
    'kp9': KEY_KP_9,
    'kpenter': KEY_KP_Enter,
}
```
* 12/13

Setting up environment using [Universe](https://openai.com/blog/universe/)

Installing instructions is [here](https://github.com/openai/universe).

Advantage: It has lots of game on the way including atari, flash and PC game. It is basically using [docker](https://www.docker.com/) to run the environment (actually it is the game image in the container) and give observations back to the agent.

![](https://github.com/wang3303/Reinforcement-learning/blob/master/vnc_driver)

Your agent is programmatically controlling a VNC client, connected to a VNC server running inside of a Docker container in the cloud, rendering a headless Chrome with Flash enabled.

Sample code is posted here.

```
import gym
import universe # register the universe environments

env = gym.make('flashgames.DuskDrive-v0')
env.configure(remotes=1) # create one flashgames Docker container
observation_n = env.reset()

while True:
  # your agent generates action_n at 60 frames per second
  action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]
  observation_n, reward_n, done_n, info = env.step(action_n)
  env.render()
```
Start your game container allocating certain ports:
```
$ docker run --privileged --cap-add=SYS_ADMIN --ipc=host \
    -p 5900:5900 -p 15900:15900 quay.io/openai/universe.flashgames
```
or you can simply use the following to automatically create a local container:
```
env.configure(remotes = 1)
```
Then you can configure your agent to connect the VNC server (port 5900 by default) and the reward/info channel (port 15900 by default):
```
env.configure(remotes='vnc://localhost:5900+15900')
```
You can connect to multiple remotes:
```
env.configure(remotes='vnc://localhost:5900+15900,vnc://localhost:5901+15901')
```
You can run your container remotely as well:
```
env.configure(remotes='vnc://your.host.here:5900+15900')
```

* 11/21

Setting up  environment for TORCS

1. xautomation (ubuntu: sudo apt-get install xautomation)

2. python, numpy, keras, gym, Tensorflow r0.10 or higher

3. [torcs](http://torcs.sourceforge.net/index.php?artid=3&name=Sections&op=viewarticle)

	install dependencies listed on that page
	
	[Software Manual](https://arxiv.org/pdf/1304.1672.pdf)

4. [gym_torcs](https://github.com/ugo-nama-kun/gym_torcs)
	
	Try changing 64 line of gym_torcs/vtorcs-RL-color/src/modules/simu/simuv2/simu.cpp to `if (isnan((float)(car->ctrl->gear)) || isinf(((float)(car->ctrl->gear)))) car->ctrl->gear = 0;` should there be problem with compiling
	
TESTING: 

1. In the first terminal,`sudo torcs` will launch the torcs. In the GUI,s select (Race --> Practice --> Configure Race) and open TORCS server by selecting Race --> Practice --> New Race. This should result that TORCS keeps a blue screen with several text information.
2. In the second terminal, `python snakeoil3_gym.py` and you shall see a demo. (change view mode by pressing F2)
![balabla](https://github.com/wang3303/Reinforcement-learning/blob/master/torcs/result.gif)
SIMPLE HOW-TO:

```
from gym_torcs import TorcsEnv

#### Generate a Torcs environment
# enable vision input, the action is steering only (1 dim continuous action)
env = TorcsEnv(vision=True, throttle=False)

# without vision input, the action is steering and throttle (2 dim continuous action)
# env = TorcsEnv(vision=False, throttle=True)

ob = env.reset(relaunch=True)  # with torcs relaunch (avoid memory leak bug in torcs)
# ob = env.reset()  # without torcs relaunch

# Generate an agent
from sample_agent import Agent
agent = Agent(1)  # steering only
action = agent.act(ob, reward, done, vision=True)

# single step
ob, reward, done, info = env.step(action)

# shut down torcs
env.end()
```
REMINDER:

To get the image of the game, you should select the Screen Resolution, and you need to select 64x64 for visual input (this immplementation only support this screen size, other screen size results the unreasonable visual information).
	
	

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










