#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt

vel = []
f = open('databases.txt', 'r')

for line in f:
    reg = re.findall(r': .*', line)
    reg[0] = reg[0][2:len(reg[0])]
    vel.append(float(reg[0]))

_, ax = plt.subplots()
ax.plot(vel, lw=2, color='#539caf', alpha=1)
ax.set_title('vel')
ax.set_xlabel('')
ax.set_ylabel('vel')
plt.show()
