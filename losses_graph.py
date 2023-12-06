import matplotlib.pyplot as plt
import numpy as np
import re

# Modified loss values without iteration 0
loss_data = """
epoch 0 iteration100, G Loss: 0.622607409954071, D Loss: 1.4010212421417236
epoch 0 iteration200, G Loss: 0.7272967100143433, D Loss: 1.386047124862671
epoch 0 iteration300, G Loss: 0.7401588559150696, D Loss: 1.397234320640564
epoch 1 iteration100, G Loss: 0.6124520897865295, D Loss: 1.4474921226501465
epoch 1 iteration200, G Loss: 0.6810575723648071, D Loss: 1.3866417407989502
epoch 1 iteration300, G Loss: 0.7303689122200012, D Loss: 1.3862459659576416
epoch 2 iteration100, G Loss: 0.7124195694923401, D Loss: 1.38873291015625
epoch 2 iteration200, G Loss: 0.7089831233024597, D Loss: 1.387303113937378
epoch 2 iteration300, G Loss: 0.874885618686676, D Loss: 1.3874375820159912
epoch 3 iteration100, G Loss: 0.3985002338886261, D Loss: 1.387665033340454
epoch 3 iteration200, G Loss: 0.7266682982444763, D Loss: 1.3862099647521973
epoch 3 iteration300, G Loss: 0.7378641963005066, D Loss: 1.3856427669525146
epoch 4 iteration100, G Loss: 0.6824491024017334, D Loss: 1.387473225593567
epoch 4 iteration200, G Loss: 0.7031933069229126, D Loss: 1.3891931772232056
epoch 4 iteration300, G Loss: 0.7208313345909119, D Loss: 1.385925531387329
epoch 5 iteration100, G Loss: 0.7209826111793518, D Loss: 1.3860039710998535
"""

# Parse loss data and extract generator and discriminator loss
lines = loss_data.strip().split('\n')
generator_loss = [float(line.split(', ')[1].split(': ')[1]) for line in lines]
discriminator_loss = [float(line.split(', ')[2].split(': ')[1]) for line in lines]

iterations = [int(match.group(1)) for match in re.finditer(r'iteration(\d+)', loss_data)]
num_iterations = len(iterations)  # Adjust this based on your needs

iterations = [100 + 100 * i for i in range(num_iterations)]


# Plotting the generator and discriminator loss
plt.figure(figsize=(10, 6))
plt.plot(iterations, generator_loss, label='Generator Loss', marker='o', linestyle='-', markersize=5)
plt.plot(iterations, discriminator_loss, label='Discriminator Loss', marker='o', linestyle='-', markersize=5)
plt.title('Generator and Discriminator Loss Over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
