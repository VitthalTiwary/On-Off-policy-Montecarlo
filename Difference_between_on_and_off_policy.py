import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

epsilon = 0.1

up, down, left, right = 0, 1, 2, 3
actions = [up, down, left, right]
action_prob = 0.25

# print("\nInitial policy:")
# policy = {(i, j): [action_prob] * len(actions) for i in range(grid_size) for j in range(grid_size)}
# for i in range(grid_size):
#     for j in range(grid_size):
#         print(f"State ({i}, {j}): {policy[(i, j)]}")


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
    try:
        index = next(i for i, value in enumerate(episode) if value[:2] == target_tuple)
        return index
    except StopIteration:
        return -1

def sum_rewards_from_index(episode, start_index):
    total_reward = 0
    for _, _, reward in episode[start_index:]:
        total_reward = gamma*total_reward + reward
    return total_reward

def on_policy(number_of_episodes):
    print("Generating optimal on policy.")
    episodes = number_of_episodes

    V = np.zeros((len(actions), grid_size, grid_size))

    states_action_pair = [((i, j), a) for i in range(grid_size) for j in range(grid_size) for a in actions]

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

    policy = {(i, j): [action_prob] * len(actions) for i in range(grid_size) for j in range(grid_size)}

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
        

    #print("\nOptimal policy after all episodes:")
    # for i in range(grid_size):
    #     for j in range(grid_size):
    #         print(f"State ({i}, {j}): {policy[(i, j)]}")
    
    #print("\nFinal Q-values ((r,c), a):\n")
    #print(V, "\n")
    
    print("Optimal on policy acquired.\n")
    return V, policy

def off_policy(number_of_episodes):
    print("Generating optimal off policy.")
    episodes = number_of_episodes

    Q = np.zeros((len(actions), grid_size, grid_size))

    states_action_pair = [((i, j), a) for i in range(grid_size) for j in range(grid_size) for a in actions]

    policy_behaviour = {(i, j): [action_prob] * len(actions) for i in range(grid_size) for j in range(grid_size)}

    policy_target = {(i, j): [action_prob] * len(actions) for i in range(grid_size) for j in range(grid_size)}

    def update_policy_greedy(Q):
        """Update target policy to be greedy w.r.t. Q-values"""
        for i in range(grid_size):
            for j in range(grid_size):
                if (i, j) == goal_state:
                    continue
                best_action = np.argmax(Q[:, i, j])
                for a in actions:
                    if a == best_action:
                        policy_target[(i, j)][a] = 1 - epsilon + epsilon / len(actions)
                    else:
                        policy_target[(i, j)][a] = epsilon / len(actions)

    returns_sum = np.zeros((len(actions), grid_size, grid_size))
    returns_count = np.zeros((len(actions), grid_size, grid_size))

    rho = 1.0
            
    for ep in range(episodes):
        episode = generate_episode(policy_behaviour)
        
        rho_cumulative = 1.0
        
        
        for s_a in states_action_pair:
            state, action = s_a
            first_occurrence_idx = find_first_occurrence_index(episode, s_a)
            
            if first_occurrence_idx != -1:
                G = sum_rewards_from_index(episode, first_occurrence_idx)
                
                pi = policy_target[state][action]
                b = policy_behaviour[state][action]
                if b > 0:
                    rho = pi / b
                else:
                    rho = 0
                
                rho_cumulative *= rho
                
                returns_count[action][state] += rho_cumulative
                returns_sum[action][state] += rho_cumulative * G
                Q[action][state] = returns_sum[action][state] / returns_count[action][state]

    
    update_policy_greedy(Q)

    #print("\nSample greedy target policy after episodes:")
    # for i in range(grid_size):
    #     for j in range(grid_size):
    #         print(f"State ({i}, {j}): {policy_target[(i, j)]}")

    # print("\nFinal Q-values ((r,c), a):\n")
    # print(Q, "\n")

    print("Optimal off policy acquired.\n")
    return Q, policy_target

def optimal_policy_check(optimal_policy_recieved):
    episodes_for_check = 10000
    #print("number of episodes for checking optimal policy:", episodes_for_check)

    V = np.zeros((4,5,5))

    policy = optimal_policy_recieved

    up, down, left, right = 0, 1, 2, 3
    actions = [up, down, left, right]
    action_prob = 0.25

    states_action_pair = []
    for row in range(5):
        for col in range(5):
            for a in actions:
                states_action_pair.append(((row, col), a))

    def generate_episode_for_check(policy):
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

    returns_sum = np.zeros((grid_size, grid_size))
    returns_count = np.zeros((grid_size, grid_size))
    avg_final_reward = 0
    avg_len_of_episode = 0

    for ep in range(episodes_for_check):
        episode = generate_episode_for_check(policy)
        #print(f"the legth of episode is: {len(episode)}")

        avg_len_of_episode = (ep*avg_len_of_episode + len(episode))/(ep + 1)

        for s_a in states_action_pair:
            ind = find_first_occurrence_index(episode, s_a)
            if ind != -1:
                total_reward = sum_rewards_from_index(episode, ind)
                #print(f"Sum of rewards from state {s_a} starting at index {ind} is {total_reward}.")
            else:
                total_reward = 0
                #print(f"State {s_a} is not in the episode.")
            s, a = s_a
            V[a][s] = (ep*V[a][s] + total_reward)/(ep + 1)
        
        final_reward = sum_rewards_from_index(episode, 0)
        #print(f"\nThe total reward gained from this episode is: {final_reward}\n")

        avg_final_reward = (avg_final_reward*ep + final_reward)/(ep + 1)
    #print(f"\nThe average reward gained from all episodes is: {avg_final_reward}\n")

    #print(V)

    print("Policy check complete.")
    return avg_final_reward, avg_len_of_episode

V_values = []
Q_values = []
E_values = []
R_off_values = []
R_on_values = []
Avg_len_on_values = []
Avg_len_off_values = []
Avg_len_diff_values = []

r = 75
m = 20

for e in range(r):
    print(e+1, ". Checking for ", m*(e+1), " episodes.\n")
    E_values.append(m*(e+1))

    #print(f"\nFor {e} episodes the result of the on policy is:")
    V, optimal_on_policy = on_policy(m*(e+1))
    V_values.append(V[0][0,0])

    #print(f"\nFor {e} episodes the result of the off policy is:")
    Q, optimal_off_policy = off_policy(m*(e+1))
    Q_values.append(Q[0][0,0])

    print("checking policies acquired.")
    R_on, Ep_len_on = optimal_policy_check(optimal_on_policy)
    R_off, Ep_len_off = optimal_policy_check(optimal_off_policy)

    Avg_len_diff = Ep_len_on - Ep_len_off

    #print(f"Average episode length for optimal policy from on policy is: {Ep_len_on}, and that from off policy is: {Ep_len_off}")

    Avg_len_on_values.append(Ep_len_on)
    Avg_len_off_values.append(Ep_len_off)
    Avg_len_diff_values.append(Avg_len_diff)
    R_on_values.append(R_on)
    R_off_values.append(R_off)

print("\nData accumulated.\n")
print("Generating graphs. \n")

plt.figure(figsize=(10, 6))
plt.plot(E_values, V_values, label="On-policy (V)", marker='o')
plt.plot(E_values, Q_values, label="Off-policy (Q)", marker='x')

plt.xlabel('Episodes (e)')
plt.ylabel('Value (V and Q)')
plt.title('On-policy vs Off-policy over episodes')
plt.legend()
plt.grid(True)
plt.savefig(f'Q_Value_Plot_{r}x{m}.png', dpi=300)

plt.show()

plt.figure()
plt.plot(E_values, R_on_values, label="On-policy (R_on)", marker='o')
plt.plot(E_values, R_off_values, label="Off-policy (R_off)", marker='x')

plt.xlabel('Episodes (e)')
plt.ylabel('Value (R_on and R_off)')
plt.title('On-policy vs Off-policy over episodes')
plt.legend()
plt.grid(True)

plt.show()

E_values = np.array(E_values)
R_off_values = np.array(R_off_values)
R_on_values = np.array(R_on_values)

slope1, intercept1 = np.polyfit(E_values, R_on_values, 1)
best_fit_line1 = slope1 * E_values + intercept1

slope2, intercept2 = np.polyfit(E_values, R_off_values, 1)
best_fit_line2 = slope2 * E_values + intercept2

plt.scatter(E_values, R_on_values, color='blue', label='Data Points R_on')
plt.scatter(E_values, R_off_values, color='orange', label='Data Points R_off')

plt.plot(E_values, best_fit_line1, color='black', label='Best Fit Line R_on')

plt.plot(E_values, best_fit_line2, color='red', label='Best Fit Line R_off')

plt.xlabel('Episodes')
plt.ylabel('Reward')
plt.title('Best Fit Lines for R_on and R_off')
plt.legend()
plt.savefig(f'Rewards_Plot_{r}x{m}.png', dpi=300)

plt.show()

plt.figure()
plt.plot(E_values, Avg_len_on_values, label="On-policy (Average Episode Length)", marker='o')
plt.plot(E_values, Avg_len_off_values, label="Off-policy (Average Episode Length)", marker='x')

plt.xlabel('Episodes (e)')
plt.ylabel('Value (Length for on and off)')
plt.title('On-policy vs Off-policy over episodes')
plt.legend()
plt.grid(True)
plt.savefig(f'Average_Length_Plot_{r}x{m}.png', dpi=300)

plt.show()

def inverse_func(x, a, b):
    return a / x + b  

E_values = np.array(E_values)
Avg_len_diff_values = np.array(Avg_len_diff_values)

params, _ = curve_fit(inverse_func, E_values, Avg_len_diff_values)

a, b = params

x_fit = np.linspace(min(E_values), max(E_values), 100)  
y_fit = inverse_func(x_fit, a, b)

plt.figure()
plt.plot(E_values, Avg_len_diff_values, label="Difference in Average Episode Length (on - off)", marker='x')

plt.plot(x_fit, y_fit, color='red', label='Best Fit Curve (1/x)')

plt.xlabel('Episodes (e)')
plt.ylabel('Difference in Length for on and off')
plt.title('Difference in Average Episode Length over Episodes')

plt.legend()
plt.grid(True)
plt.savefig(f'Average_Length_Difference_Plot_{r}x{m}.png', dpi=300)

plt.show()