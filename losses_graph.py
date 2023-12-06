import matplotlib.pyplot as plt
import numpy as np
import re

# Modified loss values without iteration 0
loss_data = """
epoch 0 iteration100, G Loss: 27.645231246948242, D Loss: 1.7421889305114746
epoch 0 iteration200, G Loss: 0.6217522621154785, D Loss: 1.7510735988616943
epoch 0 iteration300, G Loss: 1.132054090499878, D Loss: 1.7664289474487305
epoch 1 iteration100, G Loss: 0.9069512486457825, D Loss: 1.4492571353912354
epoch 1 iteration200, G Loss: 2.8514368534088135, D Loss: 1.4047290086746216
epoch 1 iteration300, G Loss: 0.7904479503631592, D Loss: 1.3965263366699219
epoch 2 iteration100, G Loss: 3.1538679599761963, D Loss: 2.059518337249756
epoch 2 iteration200, G Loss: 0.7567435503005981, D Loss: 1.447718858718872
epoch 2 iteration300, G Loss: 0.7164775133132935, D Loss: 1.4030804634094238
epoch 3 iteration100, G Loss: 0.8493850827217102, D Loss: 1.523701548576355
epoch 3 iteration200, G Loss: 1.2424650192260742, D Loss: 1.4820970296859741
epoch 3 iteration300, G Loss: 0.7453321218490601, D Loss: 1.4291741847991943
epoch 4 iteration100, G Loss: 0.7377573251724243, D Loss: 1.3864597082138062
epoch 4 iteration200, G Loss: 0.756231427192688, D Loss: 1.3824348449707031
epoch 4 iteration300, G Loss: 0.737855076789856, D Loss: 1.3935389518737793
epoch 5 iteration100, G Loss: 0.8647605776786804, D Loss: 1.4064245223999023
epoch 5 iteration200, G Loss: 0.4370872974395752, D Loss: 1.527603030204773
epoch 5 iteration300, G Loss: 0.6812267303466797, D Loss: 1.5698556900024414
epoch 6 iteration100, G Loss: 0.7604415416717529, D Loss: 1.3871439695358276
epoch 6 iteration200, G Loss: 0.7358837127685547, D Loss: 1.8417712450027466
epoch 6 iteration300, G Loss: 0.745964765548706, D Loss: 1.3893699645996094
epoch 7 iteration100, G Loss: 0.7356603145599365, D Loss: 1.3853836059570312
epoch 7 iteration200, G Loss: 0.054331351071596146, D Loss: 1.810903549194336
epoch 7 iteration300, G Loss: 0.7262622117996216, D Loss: 1.388880729675293
epoch 8 iteration100, G Loss: 0.7336065769195557, D Loss: 1.3930189609527588
epoch 8 iteration200, G Loss: 0.7030291557312012, D Loss: 1.386734962463379
epoch 8 iteration300, G Loss: 0.7273914813995361, D Loss: 1.384037971496582
epoch 9 iteration100, G Loss: 0.7173991203308105, D Loss: 1.3855931758880615
epoch 9 iteration200, G Loss: 0.6985913515090942, D Loss: 1.3871817588806152
epoch 9 iteration300, G Loss: 0.7149405479431152, D Loss: 1.386453628540039
epoch 10 iteration100, G Loss: 0.7426645755767822, D Loss: 1.3835349082946777
"""

# Parse loss data and extract generator and discriminator loss
lines = loss_data.strip().split('\n')
print(len(lines))
generator_loss = [float(line.split(', ')[1].split(': ')[1]) for line in lines]
discriminator_loss = [float(line.split(', ')[2].split(': ')[1]) for line in lines]

iterations = [int(match.group(1)) for match in re.finditer(r'iteration(\d+)', loss_data)]
num_iterations = len(iterations)  # Adjust this based on your needs

print(num_iterations)

iterations = [100 + 100 * i for i in range(num_iterations)]
print(iterations)



# Plotting the generator and discriminator loss
plt.figure(figsize=(10, 6))
plt.plot(iterations, generator_loss, label='Generator Loss', marker='o', linestyle='-', markersize=5)
plt.plot(iterations, discriminator_loss, label='Discriminator Loss', marker='o', linestyle='-', markersize=5)
plt.title('Generator and Discriminator Loss Over Cumulative Iterations')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
