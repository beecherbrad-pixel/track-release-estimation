import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_policy_map(agent, resolution=50):
    # Create a grid of market and genre heat (log-space)
    x = np.linspace(-2, 2, resolution)
    g = np.linspace(-2, 2, resolution)

    grid_x, grid_g = np.meshgrid(x, g)
    policy = np.zeros_like(grid_x)

    agent.eval()

    with torch.no_grad():
        for i in range(resolution):
            for j in range(resolution):
                state = torch.tensor([grid_x[i,j], grid_g[i,j]], dtype=torch.float32)
                # The agent output is the argmax: 0 for Wait, 1 for Release
                policy[i,j] = agent(state).argmax().item()

    # Plot the decision regions
    plt.contourf(grid_x, grid_g, policy, cmap="RdYlGn", alpha=0.3)
    
    # Add a descriptive colorbar
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(['Wait (Red)', 'Release (Green)'])
    
    # Structural Labels
    plt.xlabel("Market Heat ($\ln X_t$)")
    plt.ylabel("Genre Heat ($\ln X_{g,t}$)")
    plt.title("Optimal Release Policy Decision Boundary")
    
    # Grid for easier threshold identification
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save the figure for the notebook report
    plt.savefig('optimal_policy_map.png', dpi=300, bbox_inches='tight')
    # plt.show()

def run_diagnostics(env, agent, sims=500):
    rl_rev, base_rev, waits = [], [], []

    for _ in range(sims):
        s0 = env.reset()

        _, imm, _ = env.step(s0, 1, 0)
        base_rev.append(imm.item())

        state = s0
        age = 0

        while True:
            action = agent.act(state, epsilon=0)
            next_state, reward, done = env.step(state, action, age)

            if action == 1 or done:
                rl_rev.append(reward.item())
                waits.append(age)
                break

            state = state if next_state is None else next_state
            age += 1

    print("Baseline:", np.mean(base_rev))
    print("RL:", np.mean(rl_rev))
    print("Lift:", np.mean(rl_rev)/np.mean(base_rev) - 1)
    print("Wait:", np.mean(waits))