# import libraries
import numpy as np
from random import randrange
import matplotlib.pyplot as plt
import sys

# global parameters
size = 8  # dimension of the grid
gamma = 1  # no discount
alpha = 0.1
epsilon = 0.1
max_episode = 50000  # total number of episodes to train on
method = 1 # 1 for MC, 2 for Q-Learning
dirs = [[-1, 0], [0, 1], [1, 0], [0, -1]]  # agent's move direction: W, N, E, S
step_limit = 1000
agent_bomb_position = []  # store agent's and bomb's position in each step under optimal policy
direction = []  # store the agent move direction under optimal policy
total_return = []  # total return of each episode


# command input
size = int(sys.argv[1])
alpha = float(sys.argv[2])
epsilon = float(sys.argv[3])
max_episode = int(sys.argv[4])
method = int(sys.argv[5])

# initialize Q value function
def initQ(size):
	return np.zeros((size, size, size, size, 4))


Q = initQ(size)  # initial Q


# initialize policy
def initPai(size):
	return np.zeros((size, size, size, size, 4))


pai = initPai(size)  # initial policy


# initialize agent and bomb positions
# agent's location: (ax, ay)
# bomb's location: (bx, by)
def init_position():
	ax = 0
	bx = 0
	ay = 0
	by = 0
	while ax == bx and ay == by:
		bx = randrange(8)
		by = randrange(8)
		ax = randrange(8)
		ay = randrange(8)
	return ax, ay, bx, by
# initial position of agent and bomb

def move(ax, ay, bx, by, mode='naive'):
	#print("moving...")
	temp_q = []
	temp_pai = pai[ax][ay][bx][by]
	new_ax = ax
	new_ay = ay
	new_bx = bx
	new_by = by
	if (temp_pai[0] == 0 or temp_pai[0]==0.25):
		new_dir = randrange(4)
	else:
		rand = randrange(10)
		max_prob = max(temp_pai)
		if rand < 1:
			sub_rand = randrange(4)
			while temp_pai[sub_rand] == max_prob:
				sub_rand = randrange(4)
			new_dir = sub_rand
		else:
			new_dir = np.argmax(temp_pai)
	new_ax = ax + dirs[new_dir][0]
	new_ay = ay + dirs[new_dir][1]
	if mode == 'naive':  # reward structure 1
		if new_ax < 0 or new_ay < 0 or new_ax > 7 or new_ay > 7:  # agent stay at the same place
			new_ax = ax
			new_ay = ay
		elif new_ax == bx and new_ay == by:
			new_bx = bx + dirs[new_dir][0]
			new_by = by + dirs[new_dir][1]
		return ax, ay, bx, by, new_dir, new_ax, new_ay, new_bx, new_by, -1
	else:  # reward structure 2
		if new_ax < 0 or new_ay < 0 or new_ax > 7 or new_ay > 7:
			new_ax = ax
			new_ay = ay  # back to the same place
			return ax, ay, bx, by, new_dir, new_ax, new_ay, new_bx, new_by, -1;  # back to the same place
		elif new_ax == bx and new_ay == by:
			new_bx = bx + dirs[new_dir][0]
			new_by = by + dirs[new_dir][1]
			if new_bx < 0 or new_by < 0 or new_bx > 7 or new_by > 7:
				return ax, ay, bx, by, new_dir, new_ax, new_ay, new_bx, new_by, 10
			else:
				if (abs(new_bx - (size - 1) / 2) + abs(new_by - (size - 1) / 2) > abs(bx - (size - 1) / 2) + abs(
						by - (size - 1) / 2)):
					return ax, ay, bx, by, new_dir, new_ax, new_ay, new_bx, new_by, 1
				else:
					return ax, ay, bx, by, new_dir, new_ax, new_ay, new_bx, new_by, -1
		else:
			return ax, ay, bx, by, new_dir, new_ax, new_ay, new_bx, new_by, -1


def MCUpdate(SARS):
	G = 0
	while(len(SARS)>0):
		sub_SARS = SARS.pop();
		G = sub_SARS[9] + gamma * G
		update_q_value(sub_SARS[0],sub_SARS[1],sub_SARS[2],sub_SARS[3],sub_SARS[4],sub_SARS[5],sub_SARS[6],sub_SARS[7],sub_SARS[8],sub_SARS[9])
		update_policy(sub_SARS[0],sub_SARS[1],sub_SARS[2],sub_SARS[3])
	#print("updated")
	return G

def MCEpisode(alpha, epsilon, pai, mode='naive'):
	SARs = []
	ax, ay, bx, by = init_position()
	i = 0
	while i < 1000:
		ax, ay, bx, by, new_dir, new_ax, new_ay, new_bx, new_by, reward = move(ax, ay, bx, by, mode)
		#print("moved")
		SARs.append([ax, ay, bx, by, new_dir, new_ax, new_ay, new_bx, new_by, reward])
		if new_bx < 0 or new_by < 0 or new_bx > 7 or new_by > 7:
			break
		else:
			ax = new_ax
			ay = new_ay
			bx = new_bx
			by = new_by
			i = i + 1
	if i != 1000:
		return MCUpdate(SARs)
	else:
		return 10000  # do not count this episode

def q_learning_result_update(SARS, episode):
	returns = 0
	for i in range(len(SARS)):
		sub_SARS = SARS[i]
		returns = sub_SARS[5] + gamma * returns
		if episode == max_episode - 1:
			agent_bomb_position.append([sub_SARS[0], sub_SARS[1], sub_SARS[2], sub_SARS[3]])
			direction.append(sub_SARS[4])
	return returns


def update_q_value(ax, ay, bx, by, new_dir, new_ax, new_ay, new_bx, new_by, reward):
	current_state_action = Q[ax][ay][bx][by][new_dir]
	next_q = [0, 0, 0, 0] if new_bx < 0 or new_by < 0 or new_bx > 7 or new_by > 7 else Q[new_ax][new_ay][new_bx][
		new_by]
	new_current_state_action = current_state_action + alpha * (reward + gamma * max(next_q) - current_state_action)
	Q[ax][ay][bx][by][new_dir] = new_current_state_action


def update_policy(ax, ay, bx, by):
	if method == 1:  # MC algorithm
		q_current = Q[ax][ay][bx][by]
		max_q_dirs = []
		for i in range(0,len(q_current)):
			if(q_current[i]==max(q_current)):
				max_q_dirs.append(i)
		if(len(max_q_dirs)==4):
			pai[ax][ay][bx][by]=[0.25,0.25,0.25,0.25]
		else:
			max_q_prob = (1-epsilon)/len(max_q_dirs)
			min_q_prob = epsilon/(4-len(max_q_dirs))
			for i in range(0,4):
				if(i in max_q_dirs):
					pai[ax][ay][bx][by][i] = max_q_prob
				else:
					pai[ax][ay][bx][by][i] = min_q_prob
	else:  # Q-learning
		q_current = Q[ax][ay][bx][by]
		max_q_index = np.argmax(q_current)
		for i in range(4):
			pai[ax][ay][bx][by][i] = (1 - epsilon) if i == max_q_index else epsilon / 3

def q_learning_test(ax,ay,bx,by,mode):
	SARs = []
	i = 0
	while i < step_limit:
		ax, ay, bx, by, new_dir, new_ax, new_ay, new_bx, new_by, reward = move(ax, ay, bx, by, mode)
		SARs.append([ax, ay, bx, by, new_dir, reward])

		# update
		update_q_value(ax, ay, bx, by, new_dir, new_ax, new_ay, new_bx, new_by, reward)
		update_policy(ax, ay, bx, by)
		if new_bx < 0 or new_by < 0 or new_bx > 7 or new_by > 7:
			break
		else:  # update position
			ax = new_ax
			ay = new_ay
			bx = new_bx
			by = new_by
			i = i + 1
	if i != 1000:
		for SAR in SARs:
			print('agent location {ax}, {ay}, bomb location {bx},{by}, agent take action {dir} at directions {dirs}'.format(ax = SAR[0],ay = SAR[1],bx = SAR[2],by = SAR[3],dir=SAR[4],dirs = dirs))

def q_learning(mode, episode):
	SARs = []
	i = 0
	ax, ay, bx, by = init_position()
	while i < step_limit:
		ax, ay, bx, by, new_dir, new_ax, new_ay, new_bx, new_by, reward = move(ax, ay, bx, by, mode)
		SARs.append([ax, ay, bx, by, new_dir, reward])

		# update
		update_q_value(ax, ay, bx, by, new_dir, new_ax, new_ay, new_bx, new_by, reward)
		update_policy(ax, ay, bx, by)
		if new_bx < 0 or new_by < 0 or new_bx > 7 or new_by > 7:
			break
		else:  # update position
			ax = new_ax
			ay = new_ay
			bx = new_bx
			by = new_by
			i = i + 1

	if i != 1000:
		return q_learning_result_update(SARs, episode)
	else:
		return 10000

def learn(mode='naive'):
	returns = 0
	i = 0
	while i < max_episode:
		if(i%100==0):
			print("Evaluating episode "+str(i))
		if method == 1:
			returns = MCEpisode(alpha, epsilon, pai, mode)
		else:
			returns = q_learning(mode, i)
		if returns != 10000:
			total_return.append(returns)
		i = i+1
	return returns


def plot_episode():
	if method == 1:
		label = 'MC'
	else:
		label = 'Q-Learning'
	plt.plot(total_return, label=label)
	plt.title('The Total Return in Each Episode')
	plt.xlabel('number of episode')
	plt.ylabel('total return')
	plt.legend()
	plt.show()
	print(total_return[-1])
	print(agent_bomb_position)
	print(direction)


def main():
	learn(mode='complex')  # run learning method
	q_learning_test(5,0,4,1,mode = 'complex')
	plot_episode()  # plot returns for each episode


if __name__ == '__main__':
	main()
