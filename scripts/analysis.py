# Nathaniel Gordon, CS4100 Final Project
import json
import numpy as np

with open("data/results375.json", "r") as read_file:
    res_dict = json.load(read_file)

algs = ['Annealing', 'Genetic', 'Random']
metrics = ['Score', 'Curve Score', 'Card Score']

stats = np.zeros(())

for k in res_dict:
    for i in range(len(algs)):
        for j in range(len(metrics)):
            stats[i][j] = res_dict[k][algs[i]][metrics[j]]

num_trials = len(res_dict.keys())
print('Trial Count:', num_trials)

for j in range(len(metrics)):
    for i in range(len(algs)):
        print(algs[i]+' '+metrics[j]+':', stats[i][j]/num_trials)