
import pandas as pd
import os
import matplotlib.pyplot as plt


root = "results_gutenberg_v3"
data = {}
for file in os.listdir(root):
    if file.endswith('.py'):
        continue
    overshoot_factor = float(file.split('_')[-1][:-4])
    data[overshoot_factor] = pd.read_csv(os.path.join(root, file))['base_loss']




new_data = {}
new_data[1] = data[1]
new_data[14] = data[14]




data = new_data

print(list(data[1])[-1])
print(list(data[14])[-1])
exit()

print(data.keys())

# plt.figure(dpi=600)
# Create the plot
for key, value in data.items():
    plt.plot(value, label=key)

plt.title('Overshoot factor')
plt.xlabel('Training steps')
plt.ylabel('Loss')
plt.legend()
plt.savefig('fig.png')