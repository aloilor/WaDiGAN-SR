import matplotlib.pyplot as plt
import numpy as np
import re

with open('./losses.txt', 'r') as file:
    loss_data = file.read()

#delete blank lines
loss_data = '\n'.join(line for line in loss_data.split('\n') if line.strip())


# Parse loss data and extract generator and discriminator loss
lines = loss_data.strip().split('\n')
generator_loss = [float(line.split(', ')[1].split(': ')[1]) for line in lines]
discriminator_loss = [float(line.split(', ')[2].split(': ')[1]) for line in lines]

iterations = [int(match.group(1)) for match in re.finditer(r'iteration(\d+)', loss_data)]
num_iterations = len(iterations)  # Adjust this based on your needs

iterations = [100 + 100 * i for i in range(num_iterations)]


# Plotting the generator and discriminator loss
plt.figure(figsize=(10, 6))
plt.plot(iterations, generator_loss, label='Generator Loss', marker='o', linestyle='-', markersize=2.5)
plt.plot(iterations, discriminator_loss, label='Discriminator Loss', marker='o', linestyle='-', markersize=2.5)
plt.title('Generator and Discriminator Loss Over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
