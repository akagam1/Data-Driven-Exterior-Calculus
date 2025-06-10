import numpy as np
import matplotlib.pyplot as plt

data = np.load("../results/losses.npz")
losses = data['losses']

num_inputs = 3
assert len(losses) % num_inputs == 0, "Loss array length must be divisible by number of inputs."

losses_reshaped = losses.reshape(-1, num_inputs)
epochs = np.arange(1, losses_reshaped.shape[0] + 1)


plt.figure(figsize=(12, 6))
for i in range(num_inputs):
    plt.plot(epochs, losses_reshaped[:, i], label=f'Input {chr(65+i)}', marker='o', markersize=3)

plt.title(f"Loss per Epoch for {num_inputs} Inputs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f'../results/D2_losses.png')
plt.show()