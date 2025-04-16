import matplotlib.pyplot as plt
import numpy as np
import os

dropout_rates = [0.1, 0.3, 0.5, 0.7]
experiments = ['dagger', 'dagger_baseline', 'expert']
label_map = {
    'dagger': 'DAgger, GRU',
    'dagger_baseline': 'DAgger, MLP (baseline)',
    'expert': 'PPO Expert'
}

for i in range(4):
    DROPOUT_RATE = dropout_rates[i]
    plt.subplot(2, 2, i + 1)
    data_root = "dropout_{x}".format(x=DROPOUT_RATE)

    for exp in experiments:
        data = []
        for i in range(5):
            subroot = 'seed{x}'.format(x=i)
            path = os.path.join(data_root, subroot, '{e}_returns.npy'.format(e=exp))
            data.append(np.load(path))
        
        numpy_data = np.array(data)
        mean_data = np.mean(numpy_data, 0)
        std_data = np.std(numpy_data, 0)

        plt.plot(mean_data, label=label_map[exp])
        plt.fill_between(np.arange(mean_data.shape[0]), mean_data + std_data, mean_data - std_data, alpha = 0.15)

    plt.legend()
    plt.grid()
    plt.title('Dropout Rate = {x}'.format(x=DROPOUT_RATE))
    plt.xlabel('Episode')
    plt.ylabel('Return')

plt.show()



