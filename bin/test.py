from __future__ import division
import clstools.clstools  as tools
import clstools.simulator as sim
import numpy as np
import matplotlib.pyplot as plt

# Count the number of unique items in an array
def count_unique(keys):
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    return uniq_keys, np.bincount(bins)

#sigmas and sigmab and FRACTIONAL uncertanities
s,b,sigmas,sigmab,sK,Ntrials = 5,50,0.01,0.01,0.35,200000
muSB = 1
muB = 0

keysSB, countsSB = count_unique(sim.simulate(muSB,s,b,sigmas,sigmab,sK,Ntrials))
prSB = countsSB/Ntrials
    
keysB, countsB = count_unique(sim.simulate(muB,s,b,sigmas,sigmab,sK,Ntrials))
prB = countsB/Ntrials

print len(keysSB),len(prSB),len(keysB),len(prB)
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.stem(keysSB,prSB,markerfmt='ro',linefmt='r-') 
ax.stem(keysB,prB,markerfmt='bo',linefmt='b-')

# Compare to asymptotic test statistic distributions
# Need to bin results for this.
# Note: we shouldn't expect these to match exactly since the simulated test statistics are not exactly the same as assumed in the asymptotic formulae.
# Note 2: Should match in limit of no systematics though, I think. Not sure what the problem is.
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)

n, bins, patches = ax.hist(keysSB, weights=countsSB, bins=20, normed=True, histtype='stepfilled')
plt.setp(patches, 'facecolor', 'r', 'alpha', 0.5)

n, bins, patches = ax.hist(keysB, weights=countsB, bins=20, normed=True, histtype='stepfilled')
plt.setp(patches, 'facecolor', 'b', 'alpha', 0.5)

obsCLs, q1Asb, q1Ab, sigqsb, sigqb = tools.CLs_asymp(0.,s,b,b*sigmab,sigmas,returnextra=True)
x = np.arange(np.min(keysSB),np.max(keysB),0.5)
ySB = 1/np.sqrt(2*np.pi*sigqsb**2) * np.exp(-0.5*(x-q1Asb)**2/sigqsb**2)
yB  = 1/np.sqrt(2*np.pi*sigqb**2)  * np.exp(-0.5*(x-q1Ab)**2 /sigqb**2)

print x
print ySB
print yB

ax.plot(x,ySB,'r-')
ax.plot(x,yB,'b-')
plt.show()
 
#plt.hist(samples)
#plt.show()
