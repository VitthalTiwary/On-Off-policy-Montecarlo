import numpy as np
import random

grid_size = 5
goal_state = (4, 4)
initial_state = (0, 0)
portal_state = [(0, 3), (1, 0), (3, 3)]
kill_state = (3, 2)

reward_goal = 0
portal_reward = -3
reward_other = -1
reward_kill = -20
gamma = 0.8

episodes = 1
epsilon = 0.1

up, down, left, right = 0, 1, 2, 3
actions = [up, down, left, right]
action_prob = 0.25

V = np.zeros((len(actions), grid_size, grid_size))

states_action_pair = [((i, j), a) for i in range(grid_size) for j in range(grid_size) for a in actions]

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
        elif next_state in portal_state:
            next_state = random.choice(portal_state)
            reward = portal_reward
        else:
            reward = reward_goal if next_state == goal_state else reward_other
        
        episode.append((state, action, reward))
        state = next_state

    episode.append((state, action, reward_goal))  
    return episode

def find_first_occurrence_index(episode, target_tuple):
    for i, (state, action, _) in enumerate(episode):
        if (state, action) == target_tuple:
            return i
    return -1

def sum_rewards_from_index(episode, start_index):
    total_reward = 0
    for _, _, reward in episode[start_index:]:
        total_reward = reward + gamma * total_reward
    return total_reward

def update_policy(policy, V):
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) == goal_state:
                continue
            best_action = np.argmax(V[:, i, j])
            for a in actions:
                if a == best_action:
                    policy[(i, j)][a] = 1 - epsilon + epsilon / len(actions)
                else:
                    policy[(i, j)][a] = epsilon / len(actions)

print("\nInitial policy:")
policy = {(i, j): [action_prob] * len(actions) for i in range(grid_size) for j in range(grid_size)}
for i in range(grid_size):
    for j in range(grid_size):
        print(f"State ({i}, {j}): {policy[(i, j)]}")

# Initialize returns
returns_sum = np.zeros((len(actions), grid_size, grid_size))
returns_count = np.zeros((len(actions), grid_size, grid_size))

for ep in range(episodes):
    episode = generate_episode(policy)

    for s_a in states_action_pair:
        state, action = s_a
        first_occurrence_idx = find_first_occurrence_index(episode, s_a)
        
        if first_occurrence_idx != -1:
            G = sum_rewards_from_index(episode, first_occurrence_idx)
            returns_count[action][state] += 1
            returns_sum[action][state] += G
            V[action][state] = returns_sum[action][state] / returns_count[action][state]
    
    

# Final results

update_policy(policy, V)
    

print("\nOptimal policy after all episodes:")
for i in range(grid_size):
    for j in range(grid_size):
        print(f"State ({i}, {j}): {policy[(i, j)]}")
            
np.set_printoptions(precision=2)
print("\nFinal Q-values ((r,c), a):\n")
print(V, "\n")