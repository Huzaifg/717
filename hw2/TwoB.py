import numpy as np

# Parameters
mu = 0.5  # Switching rate from state 2 to state 1
nu = 0.2  # Switching rate from state 1 to state 2
total_time = 10000  # Total simulation time

# Initialize simulation
current_state = 1  # Start in state 1
time_in_state_1 = 0
time_in_state_2 = 0
switches = 0
current_time = 0

# Simulation loop
np.random.seed(42)  # 42 is the answer to the universe
while current_time < total_time:
    if current_state == 1:
        # Simulate time until next switch using a exponential distribution
        next_switch_time = np.random.exponential(1/nu)
        time_in_state_1 += next_switch_time
        current_state = 2
    else:
        next_switch_time = np.random.exponential(1/mu)
        time_in_state_2 += next_switch_time
        current_state = 1

    current_time += next_switch_time
    switches += 1

# Correct for the overshoot in the last switch to ensure the total time is correct
if current_state == 1:
    time_in_state_1 -= (current_time - total_time)
else:
    time_in_state_2 -= (current_time - total_time)

# Calculate expectations
prob_in_state_1 = time_in_state_1 / total_time
prob_in_state_2 = time_in_state_2 / total_time
expectation = prob_in_state_1 * 1 + prob_in_state_2 * 2

print(f"Let state 1 be 1 and state 2 be 2")
print(f"Let $\\nu$ = 0.2 be the rate of switching from state 1 to state 2 and $\\mu$ = 0.5 be the rate of switching from state 2 to state 1")
print(f"Emperical probability of being in state 1: {prob_in_state_1}")
print(f"Analytical probability of being in state 1: {mu / (nu + mu)}")
print(f"Emperical probability of being in state 2: {prob_in_state_2}")
print(f"Analytical probability of being in state 2: {nu / (nu + mu)}")
print(f"Emperical expected value: {expectation}")
print(f"Analytical expected value: {1 * mu / (nu + mu) + 2 * nu / (nu + mu)}")
