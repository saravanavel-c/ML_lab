import numpy as np
import gymnasium as gym
def train_q_learning(episodes=2000, alpha=0.1, gamma=0.99, epsilon=1.0):
    env = gym.make("Taxi-v3")

    state_size = env.observation_space.n
    action_size = env.action_space.n
    Q = np.zeros((state_size, action_size))

    epsilon_min = 0.01
    epsilon_decay = 0.995

    for episode in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            Q[state, action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
            )
            state = next_state

        # Decay epsilon after each episode
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    return Q