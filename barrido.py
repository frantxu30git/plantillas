import subprocess

P = [1, 2]
K = [1, 3, 5]
weights = ['uniform', 'distance']

for p in P:
    for k in K:
        for w in weights:
            subprocess.run(['python', 'PlantillaAComentar.py', str(p), str(k), w])
