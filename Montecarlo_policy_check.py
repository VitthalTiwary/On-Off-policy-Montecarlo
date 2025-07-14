import numpy as np
import random

grid_size = 5
goal_state = (4, 4)  
initial_state = (0, 0)  
portal_state = [(0,3), (1,0), (3,3)]
kill_state = (3,2) 

reward_goal = 0
portal_reward = -3
reward_other = -1
reward_kill = -20

gamma = 0.9
episodes = 200

V = np.zeros((4,5,5))

up, down, left, right = 0, 1, 2, 3
actions = [up, down, left, right]
action_prob = 0.25

states_action_pair = []
for row in range(5):
    for col in range(5):
        for a in actions:
            states_action_pair.append(((row, col), a))

def get_next_state(state, action):
    i, j = state
    if action == up and i > 0:
        return (i - 1, j)
    elif action == down and i < grid_size - 1:
        return (i + 1, j)
    elif action == left and j > 0:
        return (i, j - 1)
    elif action == right and j < grid_size - 1:
        return (i, j + 1)
    else:
        return state
    
def generate_episode(policy):
    episode = []
    state = initial_state
    while state != goal_state:
        action = random.choices(actions, weights=policy[state])[0]
        next_state = get_next_state(state, action)
        if next_state == kill_state:
            reward = reward_kill
            next_state = initial_state
        elif next_state in  portal_state:
            next_state = random.choice(portal_state)
            reward = portal_reward
        else:
            reward = reward_goal if next_state == goal_state else reward_other
        print(state, action, reward, "\n")
        episode.append((state, action, reward))
        state = next_state
    episode.append((state, action, reward_goal if state == goal_state else reward_other))
    return episode

def find_first_occurrence_index(lst, target_tuple):
    try:
        index = next(i for i, value in enumerate(lst) if value[:2] == target_tuple)
        return index
    except StopIteration:
        return -1

def sum_rewards_from_index(episode, start_index):
    total_reward = 0
    for _, _, reward in episode[start_index:]:
        total_reward = gamma*total_reward + reward
    return total_reward

policy = {}
grid_size = 5
states_values = [
    [ [0.05, 0.85, 0.05, 0.05], [0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.05, 0.85], [0.05, 0.05, 0.05, 0.85], [0.05, 0.85, 0.05, 0.05] ],
    [ [0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.05, 0.85], [0.05, 0.05, 0.05, 0.85], [0.05, 0.85, 0.05, 0.05] ],
    [ [0.85, 0.05, 0.05, 0.05], [0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.05, 0.85], [0.05, 0.05, 0.05, 0.85], [0.05, 0.85, 0.05, 0.05] ],
    [ [0.85, 0.05, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05], [0.85, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.85], [0.05, 0.85, 0.05, 0.05] ],
    [ [0.05, 0.05, 0.05, 0.85], [0.05, 0.05, 0.05, 0.85], [0.05, 0.05, 0.05, 0.85], [0.05, 0.05, 0.05, 0.85], [0.25, 0.25, 0.25, 0.25] ]
]

for i in range(grid_size):
    for j in range(grid_size):
        policy[(i, j)] = states_values[i][j]

returns_sum = np.zeros((grid_size, grid_size))
returns_count = np.zeros((grid_size, grid_size))
avg_final_reward = 0

for ep in range(episodes):
    episode = generate_episode(policy)
    
    for s_a in states_action_pair:
        ind = find_first_occurrence_index(episode, s_a)
        if ind != -1:
            total_reward = sum_rewards_from_index(episode, ind)
            print(f"Sum of rewards from state {s_a} starting at index {ind} is {total_reward}.")
        else:
            total_reward = 0
            print(f"State {s_a} is not in the episode.")
        s, a = s_a
        V[a][s] = (ep*V[a][s] + total_reward)/(ep + 1)
    
    final_reward = sum_rewards_from_index(episode, 0)
    print(f"\nThe total reward gained from this episode is: {final_reward}\n")

    avg_final_reward = (avg_final_reward*ep + final_reward)/(ep + 1)
print(f"\nThe average reward gained from all episodes is: {avg_final_reward}\n")

np.set_printoptions(precision=2)
print(V)