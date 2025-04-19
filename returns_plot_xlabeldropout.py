import matplotlib.pyplot as plt
import numpy as np
import os

dropout_rates = [0.1, 0.3, 0.5, 0.7]
experiments = {'dagger': [], 'dagger_baseline': [], 'expert': []}
label_map = {
    'dagger': 'DAgger, GRU',
    'dagger_baseline': 'DAgger, MLP (baseline)',
    'expert': 'PPO Expert'
}

for i in range(4):
    DROPOUT_RATE = dropout_rates[i]
    data_root = "dropout_{x}".format(x=DROPOUT_RATE)

    for exp in experiments.keys():
        data = []
        for i in range(5):
            subroot = 'seed{x}'.format(x=i)
            path = os.path.join(data_root, subroot, '{e}_returns.npy'.format(e=exp))
            data.append(np.load(path))
        
        numpy_data = np.array(data)
        experiments[exp].append(numpy_data)


    

for exp, data in experiments.items():
    mean = [np.mean(data[i]) for i in range(len(dropout_rates))]
    std = [np.std(data[i]) for i in range(len(dropout_rates))]
    plt.plot(dropout_rates, mean, label=label_map[exp], marker='o')
    plt.fill_between(dropout_rates, [mean[i] + std[i] for i in range(len(dropout_rates))], [mean[i] - std[i] for i in range(len(dropout_rates))], alpha=0.15)


plt.legend()
plt.grid()

plt.xlabel('Dropout Rate')
plt.ylabel('Return')
plt.xticks(dropout_rates)

plt.show()



