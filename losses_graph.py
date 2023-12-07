import matplotlib.pyplot as plt
import numpy as np
import re

# Modified loss values without iteration 0
loss_data = """
epoch 0 iteration100, G Loss: 0.7229467034339905, D Loss: 1.3915154933929443
epoch 0 iteration200, G Loss: 0.6351697444915771, D Loss: 1.3933753967285156
epoch 0 iteration300, G Loss: 0.7347120642662048, D Loss: 1.3861783742904663
epoch 1 iteration100, G Loss: 0.8094663619995117, D Loss: 1.3905402421951294
epoch 1 iteration200, G Loss: 0.739488959312439, D Loss: 1.4703670740127563
epoch 1 iteration300, G Loss: 0.7258837819099426, D Loss: 1.3863199949264526
epoch 2 iteration100, G Loss: 0.7310431003570557, D Loss: 1.3860599994659424
epoch 2 iteration200, G Loss: 0.7793965339660645, D Loss: 1.3877346515655518
epoch 2 iteration300, G Loss: 0.7267290949821472, D Loss: 1.385964035987854
epoch 3 iteration100, G Loss: 0.7271375060081482, D Loss: 1.385964274406433
epoch 3 iteration200, G Loss: 0.7178016304969788, D Loss: 1.3861461877822876
epoch 3 iteration300, G Loss: 0.5806469321250916, D Loss: 1.3870121240615845
epoch 4 iteration100, G Loss: 0.7232938408851624, D Loss: 1.386291265487671
epoch 4 iteration200, G Loss: 0.7990818619728088, D Loss: 1.3904483318328857
epoch 4 iteration300, G Loss: 0.7049357295036316, D Loss: 1.3862876892089844
epoch 5 iteration100, G Loss: 0.7276935577392578, D Loss: 1.3860245943069458
epoch 5 iteration200, G Loss: 0.7245003581047058, D Loss: 1.386484980583191
epoch 5 iteration300, G Loss: 0.7709107398986816, D Loss: 1.3875141143798828
epoch 6 iteration100, G Loss: 0.7672773003578186, D Loss: 1.389243483543396
epoch 6 iteration200, G Loss: 0.7211686372756958, D Loss: 1.3861615657806396
epoch 6 iteration300, G Loss: 0.7313986420631409, D Loss: 1.385573148727417
epoch 7 iteration100, G Loss: 0.725190281867981, D Loss: 1.385793924331665
epoch 7 iteration200, G Loss: 0.6946566104888916, D Loss: 1.3878734111785889
epoch 7 iteration300, G Loss: 0.7395988702774048, D Loss: 1.3864617347717285
epoch 8 iteration100, G Loss: 0.685893177986145, D Loss: 1.3856792449951172
epoch 8 iteration200, G Loss: 0.7708345651626587, D Loss: 1.3865141868591309
epoch 8 iteration300, G Loss: 0.8903787136077881, D Loss: 1.3858429193496704
epoch 9 iteration100, G Loss: 0.6704604029655457, D Loss: 1.3878719806671143
epoch 9 iteration200, G Loss: 0.7147351503372192, D Loss: 1.386151909828186
epoch 9 iteration300, G Loss: 0.750847578048706, D Loss: 1.3901691436767578
epoch 10 iteration100, G Loss: 0.6495674252510071, D Loss: 1.3876296281814575
epoch 10 iteration200, G Loss: 0.7228642106056213, D Loss: 1.3854113817214966
epoch 10 iteration300, G Loss: 0.6753832101821899, D Loss: 1.387925386428833
epoch 11 iteration100, G Loss: 0.728126585483551, D Loss: 1.3862671852111816
epoch 11 iteration200, G Loss: 0.7133610844612122, D Loss: 1.3861008882522583

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
