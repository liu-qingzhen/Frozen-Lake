# The training method of SARSA 10x10 is a little bit different from others, because I find the success-rate is not high
# for different maps.
# I use While loop instead of For loop to run the program and test the Q-table every fixed(default:100) episodes
# So the program will continue running when the agent has not learned the best policy
# Also, the agent will definitely learn the best policy finally if we keep running the program
# You can check the best policy printed on the screen

# The drawback is the figures drawn is not good-looking as it is designed for fixed episode training with For loop.


# Import Numpy, modified gym frozen-lake environment, matplotlib

import numpy as np 
from me5406_env import FrozenLakeEnv
from matplotlib import pyplot as plt
import os
import shutil

# parameters to tune


Epsilon_Decay = True
Epsilon_Decay_episode = 2000
iteration = 500
learning_rate = 0.5
gamma = 0.9
epsilon=0.8
# episode and iteration for test
test_episode = 20
test_iter = 500
# Calculate mean-reward every certain episodes
mean_reward_calc_epi = 100
map_size = 10

# To visualize the map and best policy with dictionary

mapping_policy = {0: '⇦', 1: '⇩', 2: '⇨', 3:'⇧'}
mapping_map =  {'S': '🏂', 'F': '❄️', 'H': '🕳️️', 'G':'🏁'}


# Generate the 4x4 map of the project if map_size is 4
# else, Generate random map of map_size

if map_size ==  10:
	env = FrozenLakeEnv(map_size=10)
elif map_size == 4:
	env = FrozenLakeEnv(map_name='4x4')
elif map_size == 5:
	env = FrozenLakeEnv(map_name='5x5')
else: env = FrozenLakeEnv(map_size=map_size)


# Create Q-table for environment

q_table = np.zeros((env.observation_space.n,env.action_space.n))

# Epsilon-Greedy method to generate a new action from state s
# a is the action which has the largest value of state s in Q-table

def epsilon_greedy(action,epsilon=0.3):
	p = np.random.uniform(0,1)
	if p < epsilon:
		return np.random.choice(env.action_space.n)
	else:
		return action

# Return the index of max numbers in an 1-d array
# Because np.argmax always return the index of the first largest value
# in an array, which will not return a action which has the same largest
# value in Q-table. So I rewrite a function to get the random choice of
# state-action when they have the same value in Q-table.

def maxindex(arg):
	index = []
	max = float('-inf')
	while True:
		for i in range(len(arg)):
			if arg[i] > max:
				index = [i]
				max = arg[i]
			elif arg[i] == max:
				index.append(i)
		break

	return index


# rALL: total rewards, y: total reward at each episode z: reward of each episode l: length of each episode

rAll=0
x=[]
y=[]
z=[]
l=[]

# Accumulated reward of certain episodes
# For computation of mean_reward
epi_reward = 0

# start training
success_rate = 0
epi = 0
while (success_rate < 1):
	epi += 1
	if (epi % mean_reward_calc_epi == 0):
		print('Episode: %d Mean_reward: %.3f'%(epi,epi_reward/mean_reward_calc_epi))
		epi_reward=0

		# Test every certain episodes(default:100, tune the mean_reward_calc_epi parameter)
		rewardtest = 0
		for i in range(test_episode):
			s = env.reset()
			for iter in range(test_iter):

				# choose action based on epsilon-greedy
				a = np.argmax(q_table[s,:])
				state_new,reward,done,_ = env.step(a)
				s = state_new
				if done:
					if reward == 1:
						rewardtest +=1
					break
		success_rate = rewardtest*1.0/test_episode
		print('success_rate=%.3f'%(success_rate))

	s = env.reset()	
	action = np.random.choice(maxindex(q_table[s,:]))
	if Epsilon_Decay == True:
		eps = max(0.1,epsilon-epi/Epsilon_Decay_episode)
	else: eps = epsilon

	# Choose action based on Epsilon-greedy

	a = epsilon_greedy(action,epsilon=eps)

	# max step length in each episode

	for iter in range(iteration):
		state_new,reward,done,_ = env.step(a)

		# choose action based on epsilon-greedy
		action_max = np.random.choice(maxindex(q_table[state_new,:]))
		action_new = epsilon_greedy(action_max,eps)

		# take an action and update Q-table
		q_table[s,a] = q_table[s,a] + learning_rate*(reward + gamma*(q_table[state_new,action_new]) - q_table[s,a])
		s = state_new
		a = action_new
		if done:
			if reward == 1:
				epi_reward += reward

			# add data to arrays if the episode in finished

			rAll=rAll+reward
			x.append(epi)
			y.append(rAll)
			z.append(reward)
			l.append(iter)

			break


# evaluation   same as training but not update Q-table

rewardtest = 0
for i in range(test_episode):
	s = env.reset()
	for iter in range(test_iter):

		# choose action based on epsilon-greedy
		a = np.argmax(q_table[s,:])
		state_new,reward,done,_ = env.step(a)
		if i == test_episode-1:
			env.render()
		s = state_new
		if done:
			if reward == 1:
				rewardtest +=1
			break

# print Q-tab;e
print('\nPrint_Q_table\n')
print(np.around(q_table,4))
print('\n')

# print best policy
print("Best _Policy:\n")

best_policy = np.argmax(q_table,axis = 1)
best_policy = [mapping_policy[x] if x in mapping_policy else x for x in best_policy]
best_policy = np.array(best_policy)
print(best_policy.reshape(map_size,map_size))
print('\n')
print('success_rate=%.3f'%(rewardtest*1.0/test_episode))
print('\n')

# Print map generated by the environment

print("MAP_GENERATED:\n")
map = env.desc1
map = desc = [[mapping_map[c] for c in line] for line in map]
map = np.array(map).reshape(map_size,map_size)
print(map)
print('\n')

# # path to save training records later

game_name = 'Figs_SARSA_%dx%d_Gamma=%.2f_eps=%.2f_episode=%d'%(map_size,map_size,gamma,epsilon,epi)
path = os.getcwd()+'/Training_Records/%s'%(game_name)

# if training was successful, then create file to save images of this training

success = False

if rewardtest == test_episode:
	success = True

if success:
	if os.path.exists(path):
		shutil.rmtree(path)
	os.mkdir(path)



# Print total reward

plt.xlabel("Episode(epsilon=%.2f, Epsilon_decay=True)"%(epsilon))
plt.ylabel("Total_Reward")
plt.plot(x,y)
if success:
	plt.savefig('%s/Total_Reward_Episode(epsilon=%.2f, Epsilon_decay=True).jpg'%(path,epsilon))
plt.show()

# Print reward of each episode

plt.xlabel("Episode(epsilon=%.2f, Epsilon_decay=True)"%(epsilon))
plt.ylabel("Reward_of_Each_episode")
plt.plot(x,z)
if success:
	plt.savefig('%s/Reward_of_Each_episode_Episode(epsilon=%.2f, Epsilon_decay=True).jpg'%(path,epsilon))
plt.show()

# Print step length of each episode

plt.xlabel("Episode(epsilon=%.2f, Epsilon_decay=True)"%(epsilon))
plt.ylabel("length of each episode")
plt.plot(x,l)
if success:
	plt.savefig('%s/Step_Length_(epsilon=%.2f, Epsilon_decay=True).jpg'%(path,epsilon))
plt.show()


