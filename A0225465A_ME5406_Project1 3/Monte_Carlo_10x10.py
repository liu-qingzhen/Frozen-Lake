import numpy as np
from me5406_env import FrozenLakeEnv
from matplotlib import pyplot as plt
import os
import shutil

# Epsilon-Greedy method to generate a new action from state s
# a is the action which has the largest value of state s in Q-table
def epsilon_greedy(a,eps=0.1):
    p = np.random.random()
    if p < 1 - eps:  # exploit
        return a
    else:  # explore
        return np.random.choice(env.action_space.n)

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


# parameters

# The action that each number represents
mapping_policy = {0: '⇦',1: '⇩',2: '⇨',3: '⇧'}
mapping_map = {'S': '🏂','F': '❄️','H': '🕳️','G': '🏁'}
# Parameters for training
Epsilon_Decay = False
GAMMA = 0.99
epsilon = 0.6
episode = 20000
map_size = 10
# Every n episodes, calculate the average reward
mean_reward_calc_epi = 100
# parameter for evaluation
test_episode = 100
test_iter = 200

# Path to create file for saving images
game_name = 'Figs_Monte_Carlo_%dx%d_Gamma=%.2f_eps=%.2f_episode=%d' % (map_size,map_size,GAMMA,epsilon,episode)
path = os.getcwd() + '/Training_Records/%s' % (game_name)

# I modified the gym.toytext.frozenlake environment
# By tuning the parameter of map_size to generate map of different sizes

if map_size == 4:
    env = FrozenLakeEnv(map_name='4x4')
else:
    env = FrozenLakeEnv(map_size=map_size)


# Initialization of Q-table
q_table = np.zeros((env.observation_space.n,env.action_space.n))

# Initialization of visited times of each state-action pair.
visit_times = np.zeros((env.observation_space.n,env.action_space.n))

# To save data for plotting
# rALL: total rewards, y: total reward at each episode 
# z: reward of each episode d: delta of each episode

rAll = 0
x = []
y = []
z = []
d = []

# generate random policy for the first episode
policy = np.random.choice(env.action_space.n,env.observation_space.n)

# Start training
epi_reward = 0

for epi in range(episode):
    # Calculate the average reward of certain episodes
    if (epi % mean_reward_calc_epi == 0):
        print('Mean_reward: %f' % (epi_reward / mean_reward_calc_epi))
        epi_reward = 0

    # For computation of delta
    delta = 0

    if Epsilon_Decay:
        eps = max(0.0,epsilon - epi / episode)
    else:
        eps = epsilon
    s = env.reset()
    a = epsilon_greedy(policy[s],eps=eps)

    # Generate the episode of (s,a,r) and save (State,Action,Reward)
    state_action_reward = [(s,a,0)]
    while True:

        # Take action a and get the next state, reward, wether this episode is done
        state_new,r,done,_ = env.step(a)

        # If the episode is done, save data and update (State,Action,Reward)
        if done:
            if r == 1:
                epi_reward += 1
            if (epi % mean_reward_calc_epi == 0):
                print('Episode: %d, Reward: %d' % (epi,r))
            rAll = rAll + r
            x.append(epi)
            y.append(rAll)
            z.append(r)
            state_action_reward.append((state_new,None,r))
            break

        # If this episode is not done, then choose a new action and update (State,Action,Reward)
        else:
            action = np.random.choice(maxindex(q_table[s,:]))
            a = epsilon_greedy(policy[state_new],eps)
            state_action_reward.append((state_new,a,r))

    G = 0
    state_action_return = []

    # Because our calculation of return stars from(St-1,At-1,Rt-1), so skip the first (s,a,r) data
    # which is the (St,None,Rt)
    first = True
    for s,a,r in reversed(state_action_reward):
        if first:
            first = False
        else:
            state_action_return.append((s,a,G))
        G = GAMMA * G + r
    state_action_return = reversed(state_action_return)

    # Create a set to save the visited (State,Action) pairs
    visited = set()

    # Update Q-table
    for s,a,G in state_action_return:
        if (s,a) not in visited:
            visit_times[s,a] += 1
            oldq = q_table[s,a]
            q_table[s,a] = (q_table[s,a] * (visit_times[s,a] - 1) + G) / visit_times[s,a]
            visited.add((s,a))
            delta = max(delta,np.abs(q_table[s,a] - oldq))

    # Update best policy A*
    policy[s] = np.random.choice(maxindex(q_table[s,:]))
    d.append(delta)

print('\n')

# Test if the result is right
# If the final policy is the best policy, then each time we will get reward 1
# In this operation, the Q-table will not be updated, we just take actions.
rewardtest = 0
for i in range(test_episode):
    s = env.reset()
    for iter in range(test_iter):
        a = np.argmax(q_table[s,:])
        state_new,reward,done,_ = env.step(a)
        if i == test_episode - 1:
            env.render()
            print('\n')
        s = state_new
        if done:
            if reward == 1:
                rewardtest += 1
            break
            
# Print Q-table and the best policy of each state
print("Q_Table:\n")
print(q_table)
print('\n')

# print best policy
print("Best_Policy:\n")
best_policy = np.argmax(q_table,axis=1)
best_policy = [mapping_policy[x] if x in mapping_policy else x for x in best_policy]
best_policy = np.array(best_policy)
print(best_policy.reshape(map_size,map_size))
print('\n')

# Print map generated by the environment
print("MAP_GENERATED:\n")
map = env.desc1
map = desc = [[mapping_map[c] for c in line] for line in map]
map = np.array(map).reshape(map_size,map_size)
print(map)
print('\n')

# Print succcess rate
print('success_rate=%.3f' % (rewardtest * 1.0 / test_episode))

# if training was successful, then create file to save images of this training
success = False

if rewardtest == test_episode:
    success = True

if success:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

# Plot the delta, total reward and reward for each episode

plt.xlabel("Episode(epsilon=%.2f, Epsilon_decay=True)" % (epsilon))
plt.ylabel("Total_Reward")
plt.plot(x,y)
if success:
    plt.savefig('%s/Total_Reward_Episode(epsilon=%.2f, Epsilon_decay=True).jpg' % (path,epsilon))
plt.show()

plt.xlabel("Episode(epsilon=%.2f, Epsilon_decay=True)" % (epsilon))
plt.ylabel("Reward_of_Each_episode")
plt.plot(x,z)
if success:
    plt.savefig('%s/Reward_of_Each_episode_Episode(epsilon=%.2f, Epsilon_decay=True).jpg' % (path,epsilon))
plt.show()

plt.xlabel("Episode(epsilon=%.2f, Epsilon_decay=True)" % (epsilon))
plt.ylabel("Delta")
plt.plot(x,d)
if success:
    plt.savefig('%s/Delta_Episode(epsilon=%.2f, Epsilon_decay=True).jpg' % (path,epsilon))
plt.show()
