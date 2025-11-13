import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define the input range for u^k
u = np.linspace(-2, 2, 200)

# Define the functions based on the table
def uncoordinated(u):
    return np.zeros_like(u)

def gaussian(u):
    # Plot a sample from N(0, 0.33)
    np.random.seed(0)
    return np.random.normal(0, 0.33, size=len(u))

def cubic(u):
    y = 0.33 * (u / 0.9)**3
    return np.clip(y, -3, 3)

def cubic_restricted(u):
    y = 0.33 * (u / 0.9)**3
    return np.clip(y, -1, 3)

def step(u):
    return np.where(u >= 0.9, 0.33, np.where(u <= -0.9, -0.33, 0.0))

# Compute values
lambda_uncoordinated = uncoordinated(u)
lambda_gaussian = gaussian(u)
lambda_cubic = cubic(u)
lambda_cubic_restricted = cubic_restricted(u)
lambda_step = step(u)

data = pd.DataFrame({
    "u": u,
    "uncoordinated": lambda_uncoordinated,
    "gaussian": lambda_gaussian,
    "cubic": lambda_cubic,
    "cubic_restricted": lambda_cubic_restricted,
    "step": lambda_step,
})

# Set up the plot
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each function
sns.lineplot(data, x="u", y="uncoordinated", label='uncoordinated', ax=ax, linewidth=2)
sns.scatterplot(data, x="u", y="gaussian", label='gaussian', alpha=0.3, ax=ax)
sns.lineplot(data, x="u", y="cubic", label='cubic', ax=ax, linewidth=2)
sns.lineplot(data, x="u", y="cubic_restricted", label='cubic_restricted',
             linestyle='--', ax=ax, linewidth=2)
sns.lineplot(data, x="u", y="step", label='step', linewidth=2, ax=ax)

# Labels and legend
plt.title('Applied Temporal Resolutions')
plt.xlabel('$u$')
plt.ylabel('$\lambda$')
plt.legend()
plt.axhline(0, color='gray', linewidth=0.5, linestyle='--')
plt.tight_layout()

# Show plot
plt.show()