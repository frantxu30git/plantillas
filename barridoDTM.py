import subprocess

P = [1, 2]
K = [1, 3, 5]
weights = ['uniform', 'distance']
s = 'a'
for p in P:
    for k in K:
        for w in weights:
            subprocess.run(['python', 'plantillaDTmulticlass.py', str(p), str(k),w,s])