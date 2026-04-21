import torch
import torch.nn.functional as F
import copy
from replay_buffer import ReplayBuffer

class DQNTrainer:
    def __init__(self, env, agent, gamma=0.99, lr=1e-4):
        self.env = env
        self.agent = agent
        self.target = copy.deepcopy(agent)
        self.target.load_state_dict(agent.state_dict())
        self.target.eval()

        self.optimizer = torch.optim.Adam(agent.parameters(), lr=lr)
        self.buffer = ReplayBuffer()
        self.gamma = gamma

    def update_target(self):
        self.target.load_state_dict(self.agent.state_dict())

    def run_episode(self, epsilon=0.1):
        state = self.env.reset()
        age = 0
        done = False

        while not done:
            action = self.agent.act(state, epsilon)
            next_state, reward, done = self.env.step(state, action, age)

            reward = self.env.reward(state)/100000 if action == 1 else 0.0
            

            self.buffer.push((
                state,
                action,
                reward,
                next_state if next_state is not None else torch.zeros(2),
                done
            ))

            state = state if next_state is None else next_state
            age += 1

    def update(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        states = torch.stack(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)

        # Q(s,a)
        q_values = self.agent(states).gather(1, actions.unsqueeze(1)).squeeze()

        # target Q
        with torch.no_grad():
            next_q = self.target(next_states).max(1)[0]

        # reward normalization stability guard
        rewards = torch.clamp(rewards, -5, 5)

        targets = rewards + self.gamma * next_q * (1 - dones)

        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def train(self, episodes=5000, batch_size=64):
        for ep in range(episodes):
            self.run_episode(epsilon=0.1)
            loss = self.update(batch_size)

            if ep % 100 == 0:
                self.update_target()

            if ep % 500 == 0:
                print(f"Episode {ep}, Loss: {loss}")