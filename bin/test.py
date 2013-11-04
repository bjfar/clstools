from __future__ import division
import clstools.simulator as sim
import numpy as np
import matplotlib.pyplot as plt

# Count the number of unique items in an array
def count_unique(keys):
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    return uniq_keys, np.bincount(bins)

s,b,sigmas,sigmab,Ntrials = 5,10,2,3,10000
muSB = 1
muB = 0

keysSB, countsSB = count_unique(sim.simulate(muSB,s,b,sigmas,sigmab,Ntrials))
prSB = countsSB/Ntrials
    
keysB, countsB = count_unique(sim.simulate(muB,s,b,sigmas,sigmab,Ntrials))
prB = countsB/Ntrials

print len(keysSB),len(prSB),len(keysB),len(prB)
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.stem(keysSB,prSB,markerfmt='ro',linefmt='r-') 
ax.stem(keysB,prB,markerfmt='bo',linefmt='b-')
plt.show()
 
plt.hist(samples)
plt.show()
