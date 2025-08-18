import torch
import torch.nn as nn
import gymnasium as gym


class PolicyNetwork(nn.Module):
    """Policy network - outputs action logits"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)


class ValueNetwork(nn.Module):
    """Value network - outputs state value estimate"""

    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Single output for value

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x).squeeze()  # Remove extra dimension


# Environment setup
env = gym.make("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Networks and optimizers
policy_net = PolicyNetwork(state_dim, action_dim)
value_net = ValueNetwork(state_dim)

policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.001)
value_optimizer = torch.optim.Adam(value_net.parameters(), lr=0.001)

num_updates = 200

for update in range(num_updates):
    episodes_per_update = 100

    # Policy training
    policy_optimizer.zero_grad()
    total_policy_loss = 0

    # Value training
    all_states = []
    all_returns_to_go = []
    episode_returns = []

    for episode in range(episodes_per_update):
        states = []
        log_probs = []
        rewards = []

        obs, _ = env.reset()
        done = False

        while not done:
            state_tensor = torch.tensor(obs, dtype=torch.float32)
            states.append(state_tensor)

            logits = policy_net(state_tensor)
            probs = torch.softmax(logits, dim=0)
            action = torch.multinomial(probs, num_samples=1)
            log_probs.append(torch.log(probs[action]))

            obs, reward, term, trunc, _ = env.step(action.item())
            rewards.append(reward)
            done = term or trunc

        # Compute returns-to-go and advantages for this episode
        returns_to_go = []
        for i in range(len(rewards)):
            returns_to_go.append(sum(rewards[i:]))

        # Get value predictions for this episode (for advantages)
        with torch.no_grad():
            states_tensor = torch.stack(states)
            values = value_net(states_tensor)
            advantages = torch.tensor(returns_to_go, dtype=torch.float32) - values

        # Accumulate policy loss for this episode (like REINFORCE)
        episode_policy_loss = 0
        for i in range(len(log_probs)):
            episode_policy_loss += log_probs[i] * advantages[i]

        total_policy_loss += episode_policy_loss

        # Store data for value network training
        all_states.extend(states)
        all_returns_to_go.extend(returns_to_go)
        episode_returns.append(sum(rewards))

    # Update policy (average over episodes, like REINFORCE)
    avg_policy_loss = -total_policy_loss / episodes_per_update
    avg_policy_loss.backward()
    policy_optimizer.step()

    # Update value network (batched)
    value_optimizer.zero_grad()
    states_tensor = torch.stack(all_states)
    returns_tensor = torch.tensor(all_returns_to_go, dtype=torch.float32)
    value_predictions = value_net(states_tensor)
    value_loss = nn.MSELoss()(value_predictions, returns_tensor)
    value_loss.backward()
    value_optimizer.step()

    if update % 10 == 0 or update == num_updates - 1:
        avg_return = sum(episode_returns) / len(episode_returns)
        min_return = min(episode_returns)
        max_return = max(episode_returns)
        print(
            f"Update {update}: Avg return = {avg_return:.1f}, "
            f"Min return = {min_return:.1f}, "
            f"Max return = {max_return:.1f}, "
            f"Value loss = {value_loss.item():.3f}"
        )
