# imports
import os
import time
import numpy as np
# %tensorflow_version 1.x
import tensorflow as tf
print(tf.__version__)
import gym
import random
from collections import deque
import matplotlib.pyplot as plt
# choose a GPU card
os.environ['CUDA_VISIBLE_DEVICES']="0"

# Set seed for tensorflow
SEED=2
tf.set_random_seed(SEED)
GYM_SEED=8
#
session_conf = tf.ConfigProto(
      intra_op_parallelism_threads=3,
      inter_op_parallelism_threads=3)

env = gym.make("MsPacman-v0")
obs = env.reset()

mspacman_color = 210 + 164 + 74
def preprocess(obs):
    img = obs[1:176:2, ::2] # crop and downsize
    img = img.sum(axis=2) # to greyscale
    img[img==mspacman_color] = 0 # Improve contrast
    img = (img // 3 - 128).astype(np.int8) # normalize from -128 to 127
    return img.reshape(88, 80, 1)

# This is now exactly the same as Ghani's CNN settings
# need to adjust to 80**)
learning_rate=0.001
# which is height? Which is width?
state_size=[88,80,1]
action_size=env.action_space.n
n_outputs=action_size
input_height=88
input_width=80
input_channels=1


# Define a Q network with input size of [88,80], and an output size of size of action space 9
class DQN(object):
	def __init__(self, state_size, action_size, learning_rate, name='online_q_network'):
		self.state_size = state_size
		self.action_size = action_size
		self.learning_rate = learning_rate
		self.name = name

		with tf.variable_scope(name):
			with tf.name_scope("inputs"):
				# Jieda note: state is preprocessd 80*80*1 array
				self.inputs = tf.placeholder(tf.float32, [None, *state_size], name="inputs")

			with tf.name_scope("conv1"):
				self.conv1 = tf.layers.conv2d(
					inputs=self.inputs, filters=32, kernel_size=[8, 8], strides=4,
					kernel_initializer=tf.variance_scaling_initializer(scale=2),
					padding="VALID", activation=tf.nn.relu, use_bias=False, name='conv1')

				self.conv1_out = tf.nn.relu(self.conv1, name='conv1_out')

			with tf.name_scope("conv2"):
				self.conv2 = tf.layers.conv2d(
					inputs=self.conv1_out, filters=64,
					kernel_size=[4, 4], strides=[2, 2], padding="VALID",
					kernel_initializer=tf.variance_scaling_initializer(scale=2),
					activation=tf.nn.relu, use_bias=False, name='conv2')

				self.conv2_out = tf.nn.relu(self.conv2, name='conv2_out')

			with tf.name_scope("conv3"):
				self.conv3 = tf.layers.conv2d(
					inputs=self.conv2_out, filters=64,
					kernel_size=[3, 3], strides=[1, 1], padding="VALID",
					kernel_initializer=tf.variance_scaling_initializer(scale=2),
					name="conv3")

				self.conv3_out = tf.nn.relu(self.conv3, name='conv3_out')

			with tf.name_scope("flatten"):
				self.flatten = tf.contrib.layers.flatten(self.conv3_out)

			with tf.name_scope("fc1"):
				self.fc1 = tf.layers.dense(inputs=self.flatten,
				                           units=512, activation=tf.nn.relu,
				                           kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc1")

			with tf.name_scope("fc1"):
				self.fc2 = tf.layers.dense(inputs=self.flatten,
				                           units=512, activation=tf.nn.relu,
				                           kernel_initializer=tf.contrib.layers.xavier_initializer(), name="fc2")

			with tf.name_scope("outputs"):
				self.outputs = tf.layers.dense(inputs=self.fc2,
				                               units=action_size,
				                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
				                               activation=None)
		# Output is the approximated Action Values Q(s,a), so we don't need any activation

	def get_outputs(self):
		return self.outputs

	def get_weights(self, scope_name):
		# give all the weights of that network
		trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
		# create a dictionary to contain the values of the network weights
		trainable_vars_by_name = {var.name[len(scope_name):]: var
		                          for var in trainable_vars}
		return trainable_vars_by_name

#
tf.reset_default_graph()
X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels])
# initialize the two Q network
online_q = DQN(state_size, action_size, learning_rate, 'online_q_network')
target_q = DQN(state_size, action_size, learning_rate, 'target_q_network')
# get the output and weights from online q network
online_q_values = online_q.get_outputs()
online_q_weights = online_q.get_weights(scope_name='online_q_network')
# # get the output and weights from target q network
target_q_values = target_q.get_outputs()
target_q_weights = target_q.get_weights(scope_name='target_q_network')
# define the operation to copy the online network weights to target_q_network weights
copy_ops = [target_var.assign(online_q_weights[var_name]) for var_name, target_var in target_q_weights.items()]
#
copy_online_to_target = tf.group(*copy_ops)

## define the train operation
with tf.variable_scope("train"):
	# define training operation
	input_action = tf.placeholder(tf.int32, shape=[None])
	y = tf.placeholder(tf.float32, shape=[None, 1])
	# Get the Q values for the input_action
	q_value = tf.reduce_sum(online_q_values * tf.one_hot(input_action, action_size), axis=1, keepdims=True)
	# compute the error between lable y and q_value from online_q_network approximation
	# Note that lable y is computed using the target_q_network
	error = tf.abs(y - q_value)
	clipped_error = tf.clip_by_value(error, 0, 1)
	linear_error = 2 * (error - clipped_error)
	loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

	# global_step is used to keep track of number of training steps completed
	global_step = tf.Variable(0, trainable=False, name='global_step')
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08,
	                                   use_locking=False, name='Adam')
	training_op = optimizer.minimize(loss, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

replay_memory_size = 200000


# replay_memory = deque([],maxlen=replay_memory_size)

class ReplayBuffer:
	def __init__(self, maxlen):
		self.maxlen = maxlen
		self.buf = np.empty(shape=maxlen, dtype=np.object)
		self.index = 0
		self.length = 0

	def append(self, data):
		self.buf[self.index] = data
		self.length = min(self.length + 1, self.maxlen)
		self.index = (self.index + 1) % self.maxlen

	def sample(self, batch_size):
		# sample without replacement
		indices = np.random.permutation(self.length)[:batch_size]
		return self.buf[indices]

# create an instance of ReplayBuffer
replay_buffer = ReplayBuffer(replay_memory_size)


def sample_replay_buffer(batch_size):
	cols = [[], [], [], [], []]  # state, action, reward, next_state, continue
	for memory in replay_buffer.sample(batch_size):
		for col, value in zip(cols, memory):
			col.append(value)
	cols = [np.array(col) for col in cols]
	return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)

eps_min = 0.1
eps_max = 1.0
eps_decay_steps = 2000000
def epsilon_greedy(q_values, step):
    # Note: we gradually decrease epsilon, we explore more in the beginning, less towards later
    epsilon = max(eps_min, eps_max-(eps_max-eps_min)*step/eps_decay_steps)
    if np.random.rand()<epsilon:
        return np.random.randint(n_outputs)
    else:
        return np.argmax(q_values)

# This is the number of training steps
n_steps = 4000000
# In the very beginning, First have 10000 samples in the replay buffer
training_start=10000
# Before each training step, we run 4 episodes and add those samples to replay buffer
training_interval=4
#
save_steps=1000
copy_steps = 2500 # copy the online network parameters to target network every 10000 steps
discount_rate=0.99
skip_start=90
batch_size=50
iteration=0
done=True
# initialize the tracking statistics
max_episode_reward_so_far=0
total_max_q=-float('inf')
mean_reward=0
total_reward=0
game_length=0
game_counter=0

checkpoint_path="./mspacman_dqn_run3_nohup.ckpt"

with tf.Session(config=session_conf) as sess:
	if os.path.isfile(checkpoint_path + ".index"):
		saver.restore(sess, checkpoint_path)
	else:
		init.run()
		# to make sure they have the same values to begin with
		# otherwise, different seeds will result in different initialized network weights
		copy_online_to_target.run()
	# intialize replay buffer
	# replay_buffer = ReplayBuffer(replay_memory_size)
	# initialize deque for keeping the last 30 episode rewards
	#   all_rwds_deque = deque([0]*30)
	# initialize 2 lists to store the final training results
	final_each_episode_rwds = []
	final_mean_max_q = []
	# start training
	# Training loop :
	# (1) collect M data points, add to replay buffer (play 4 episodes of game, add those to replay buffer)
	# (2) copy online network parameter to target network
	# (3) sample a batch from replay buffer, do the online network update (batch size:50)
	while True:
		step = global_step.eval()
		# if global training steps have meet the maximum, we will stop training
		if step >= n_steps:
			break
		# One iteration is just one sample generation !!!
		iteration += 1
		if done:
			obs = env.reset()
			for skip in range(skip_start):  # skip the beginning of each game
				obs, reward, done, info = env.step(0)  # take action 0
			state = preprocess(obs)

		state = state.reshape(1, 88, 80, 1)
		state_to_rb = state.reshape(88, 80, 1)

		# online network evaluates what to do
		q_values = online_q_values.eval(feed_dict={online_q.inputs: state})
		# q_values returns array of 4 values corresponding to 4 actions
		# for each step in the game, we record the max of q_values
		action = epsilon_greedy(q_values, step)

		# online network plays
		obs, reward, done, info = env.step(action)
		next_state = preprocess(obs)
		next_state = next_state.reshape(1, 88, 80, 1)
		next_state_to_rb = next_state.reshape(88, 80, 1)

		# Add this sample to the replay buffer
		replay_buffer.append((state_to_rb, action, reward, next_state_to_rb, 1.0 - done))
		# pass "next_state" to "state"
		state = next_state
		state_to_rb = next_state_to_rb

		# compute statistics to track training progress
		# track the maximum episode reward obtained so far, and the average episode reward
		total_reward += reward
		total_max_q += q_values.max()
		game_length += 1
		if done:
			game_counter += 1
			mean_max_q = total_max_q / game_length
			new_episode_reward = total_reward
			# print stats
			print("Iteration %d\tTraining step %d/%d  Mean Max Q %f last episode reward %d Game counts %d game length "
				"%d" % (iteration, step, n_steps, mean_max_q, total_reward, game_counter, game_length))
			# append the results to the final output
			final_each_episode_rwds.append(new_episode_reward)
			final_mean_max_q.append(mean_max_q)
			# reset the counters
			total_max_q = 0.0
			game_length = 0
			total_reward = 0

		# if number of samples are less than training_start(10000), or it is not a multiple of 4, we will NOT do
		# training
		if iteration < training_start or iteration % training_interval != 0:
			continue

		#     if iteration % 1000 == 0: # print stats every 100 training steps
		#       print("Iteration %d\tTraining step %d/%d  Mean Max Q %f episode total reward %d  Max episode reward so
		#       far %d  Game length %d  Game counts %d" %(iteration, step, n_steps, mean_max_q, total_reward,
		#       max_episode_reward_so_far, game_length, game_counter))
		# otherwise, if iteration is more than 10,000 and it is a multiple of 4, we will update the network
		# result=sample_replay_buffer(batch_size)
		X_state_val, X_action_val, rewards, X_next_state_val, continues = sample_replay_buffer(batch_size)
		#
		next_q_values = target_q_values.eval(
			feed_dict={target_q.inputs: X_next_state_val}
		)
		max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
		y_val = rewards + continues * discount_rate * max_next_q_values

		# train online q network
		_, loss_val = sess.run([training_op, loss],
		                       feed_dict={online_q.inputs: X_state_val, input_action: X_action_val, y: y_val})

		# copy online q network to target network
		if step % copy_steps == 0:
			copy_online_to_target.run()

		# save dqn regularly
		if step % save_steps == 0:
			saver.save(sess, checkpoint_path)
			# save output to text
			with open('pacman_run3_each_episode_reward.txt', 'w') as file:
				file.write('%s\n' % final_each_episode_rwds)
			with open('pacman_run3_mean_max_q.txt', 'w') as file:
				file.write('%s\n' % final_mean_max_q)
