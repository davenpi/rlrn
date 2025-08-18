import torch
import torch.nn as nn
import gymnasium as gym


class LogitsNetwork(nn.Module):
    """
    A simple feedforward neural network that takes a state and outputs a logits vector
    over actions.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(LogitsNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        return self.fc2(x)


# Create the environment
env = gym.make("CartPole-v1")

# Get the state and action dimensions
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

logits_network = LogitsNetwork(state_dim, action_dim)
lr = 0.001
optimizer = torch.optim.Adam(logits_network.parameters(), lr=lr)

num_updates = 200

for update in range(num_updates):
    episodes_per_update = 100
    optimizer.zero_grad()

    total_psuedo_loss = 0

    returns = []
    for episode in range(episodes_per_update):
        obs, info = env.reset()
        trunc = False
        term = False
        ret = 0
        episode_log_probs = 0
        while not trunc and not term:
            logits = logits_network(torch.tensor(obs, dtype=torch.float32))
            probs = torch.softmax(logits, dim=0)
            action = torch.multinomial(probs, num_samples=1)
            episode_log_probs += torch.log(probs[action])
            obs, rew, term, trunc, info = env.step(action.item())
            ret += rew

        returns.append(ret)
        total_psuedo_loss += episode_log_probs * ret

    avg_psuedo_loss = -total_psuedo_loss / episodes_per_update
    avg_psuedo_loss.backward()
    optimizer.step()

    if update % 10 == 0 or update == num_updates - 1:
        print(
            f"Update {update}: Avg return = {sum(returns) / len(returns):.1f}, "
            f"Min = {min(returns)}, Max = {max(returns)}"
        )
