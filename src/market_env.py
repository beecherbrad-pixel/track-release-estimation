import torch
import numpy as np

class MarketEnv:
    def __init__(self, phi, sigma, beta_m, beta_g, lambd, delta, eta_mean):
        self.phi = torch.tensor(phi, dtype=torch.float32)
        self.sigma_chol = torch.tensor(np.linalg.cholesky(sigma), dtype=torch.float32)

        self.beta_m = float(beta_m)
        self.beta_g = float(beta_g)
        self.lambd = float(lambd)
        self.delta = float(delta)
        self.eta = float(eta_mean)

        days = np.arange(365)
        self.lifetime_mult = np.sum((delta ** days) * np.exp(-self.lambd * days))

    def reset(self):
        return torch.tensor([0.0, 0.0], dtype=torch.float32)

    def reward(self, state):
        # state is [log_xm, log_xg]
        log_xm, log_xg = state[0], state[1]
        
        # structural form: y = exp(eta + beta_m * x_m + beta_g * x_g)
        log_potential = self.eta + self.beta_m * torch.exp(log_xm) + self.beta_g * torch.exp(log_xg)
        
        # Reward is the level of streams (always positive)
        return torch.exp(log_potential) * self.lifetime_mult
        return raw_reward / 1e5

    def step(self, state, action, age):
        if action == 1: # Release
            return None, self.reward(state), True
        
        # Wait: Evolve market state
        noise = torch.randn(2) @ self.sigma_chol.T
        next_state = self.phi @ state + noise
        
        return next_state, torch.tensor(0.0), False