#!/bin/python

"""CLs limit setting tools

"""
from __future__ import division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle as p
import scipy.interpolate as interp
import scipy.integrate as integ
import minuit
import pymc
import simulator as sim
from numpy.lib import recfunctions
from scipy.stats import norm, poisson

import pylab

#global options that may be desired to be tweaked

font = {'family' : 'serif',
        'weight' : 'normal',  #'bold'
        'size'   : 16,
        }
mpl.rc('font', **font)
mpl.rcParams['lines.solid_joinstyle'] = 'bevel' #is 'round' by default, but this fails to render correctly in output eps files
mpl.rcParams['lines.dash_joinstyle'] = 'bevel' #is 'round' by default, but this fails to render correctly in output eps files

#=======================================================================
# Helper functions
#=======================================================================

def logpoissonlike(n,mu):
    """Likelihood function for single bin counting experiment
        Args:
        n - observed number of counts
        mu - expected number of counts
    """
    if n==None: return -1e300
    return pymc.poisson_like(n,mu)
    
#str representation of float for use as dictionary key (to avoid floating
#point rounding errors)
def f2s(float):
    return "%.8g" % float
    
#check if x can be represented as a floating point number
def isfloat(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

def extrap1d(interpolator):
    """wraps interpolating function and gives a flat extrapolation outside ranges of input data"""
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]
        elif x > xs[-1]:
            return ys[-1]
        else:
            return interpolator(x)

    #def ufunclike(xs):
    #    return np.array(map(pointwise, np.array(xs)))
    vfunc = np.vectorize(pointwise)  #seems slow, but the other way didn't seem to work...

    #return ufunclike
    return vfunc

# Count the number of unique items in an array
def count_unique(keys):
    uniq_keys = np.unique(keys)
    bins = uniq_keys.searchsorted(keys)
    return uniq_keys, np.bincount(bins)

def grid2duneven(coords,zvals,gridXYvals,unevenaxis=0,tol=0):
   """2d linear interpolation with even grid in one direction and uneven grid in other direction
      Casts data onto a regular grid for use in plotting.
   Actually the grid does not need to be even; the points in the "unevenaxis" direction just have to be almost colinear. The terms I picked are not very clear, sorry.
numpy where
   tol - units of tolerance to allow for collection of data into bins in the (almost) even direction i.e. when interpolating in the uneven direction, points in the other direction within this tolerance will be consider colinear.
   """
   # Collect points into columns for each step along even direction    
   evenaxis = 1-unevenaxis
   bins = set(coords[:,evenaxis])
   evencoords = [] #we will take the median value in each bin as the bin coordinate
   resampled = [] #resampled data in colinear direction stored here
   bins = sorted(list(bins))
   print bins
   notdonemask = np.ones(len(coords),dtype=bool)
   for val in bins:
      print val+tol, val-tol
      matchmask = (coords[:,evenaxis] <= val+tol) & (coords[:,evenaxis] >= val-tol)
      getmask = matchmask & notdonemask
      if sum(getmask)<=1: continue   # can't interpolate if only 1 match
      notdonemask = (~matchmask) & notdonemask
      # get interpolating function for the z values at these points along the uneven direction
      sorti = np.argsort(coords[getmask,unevenaxis])
      print sorti
      print coords[getmask,unevenaxis][sorti]
      print zvals[getmask][sorti]
      #ifunc = extrap1d(interp.interp1d(coords[getmask,unevenaxis][sorti],
      #                                  zvals[getmask][sorti],
      #                                  bounds_error=False,fill_value=0))
      # I don't think there is any need for extrap?
      ifunc = interp.interp1d(coords[getmask,unevenaxis][sorti],
                                        zvals[getmask][sorti],
                                        bounds_error=False,fill_value=0)
 
      # resample data to regular grid in requested direction
      evencoords += [np.median(coords[getmask,evenaxis])]
      resampled += [ifunc(gridXYvals[unevenaxis])]

   # Now interpolate the regular resampled data in the remaining direction 
   resampled = np.array(resampled)
   row_resampled = []
   print "other direction..."
   for row in resampled.T:
      print evencoords
      print row
      print len(evencoords), len(row)
      #row_ifunc = extrap1d(interp.interp1d(evencoords,row,bounds_error=False,fill_value=0))
      row_ifunc = interp.interp1d(evencoords,row,bounds_error=False,fill_value=0)
      row_resampled += [row_ifunc(gridXYvals[evenaxis])]
   
   #output grid
   return np.array(row_resampled)   

def sqrm(x): return np.sqrt(np.abs(x))

# Asympotical formula for CLs test statistic calculations
def q(mu,s): return (1.-2.*mu)/s**2

def qcomb(mu1,s1,mu2,s2,rho):
    return 1./(1.-rho) * ( q(mu1,s1) + q(mu2,s2) - 2.*rho*(1.-mu1-mu2)/(s1*s2) )

rv = norm() #standard normal
def CLs(qobs,MAGqA):
    """This makes the assumption that q_s+b and q_b have the same variance!
    Otherwise the psb and pb values need slightly different qobs and qA values.
    As a result, qA_s+b = -qA_b, so the MAGNITUDE should be supplied here. We
    will take the absolute value anyway to make sure."""
    qA = np.abs(MAGqA)
    psb = 1 - rv.cdf(0.5*(qobs + qA)/np.sqrt(qA)) 
    pb  =     rv.cdf(0.5*(qobs - qA)/np.sqrt(qA)) 
    return psb/(1-pb)

def CLs_2(qobs,qAsb,qAb):
    """As above but doesn't make the small signal approximation"""
    psb = 1 - rv.cdf( 0.5*(qobs - qAsb)/np.sqrt(np.abs(qAsb)) ) 
    pb  =     rv.cdf( 0.5*(qobs - qAb)/np.sqrt(np.abs(qAb))   ) 
    return psb/(1-pb)

def CLs_asymp(n,s,b,sigb=None,fsigs=0.,returnextra=False):
    """Obtain CLs values for test signal hypotheses using asymptotic approximations
    n - number of signal candidates observed
    s - predicted mean number of signal events
    sigb - standard devitation of background (in units of event number)
      (i.e. includes the poisson uncertainty: sigb**2 = b + sig_extra**2) 
    fsigs - standard deviation of signal rate (as fraction of s)
      (i.e. does not include poisson uncertainty; is just uncertainty on signal prediction)
    """
    #b = self.b
    #sigb = b*self.bstat
    #fsigs = self.ssys
    #print s
    #print sigb

    #if no background standard deviation provided, assume poisson only
    if sigb==None: sigb = np.sqrt(b)
    
    ## mu = (n-b)/s  # Maximum likelihood estimator for mu, i.e. \hat{\mu}
    ## sigmusb = np.sqrt(1*s + (1*s*fsigs)**2 + sigb**2)/s  #mu'=1
    ## sigmub = np.sqrt(0*s + (0*s*fsigs)**2 + sigb**2)/s   #mu'=0
    ## #print sigmub
    ## # Determine number of "sigmas" the observation is above the background:
    ## #print "significance (units of sigma_b) = ", (mu/sigmub)[0]
    ## 
    ## q1sb = (1-2*mu)/sigmusb**2
    ## q1b  = (1-2*mu)/sigmub**2
    ##
    ## q1Asb = -1/sigmusb**2   #mu'=1 (sign doesn't actually matter, gets killed by abs)
    ## q1Ab  = +1/sigmub**2   #mu'=0 (sigmusb~sigmub so q1Asb ~ -q1Ab)
    ## 
    ## sigqsb = 2/sigmusb
    ## sigqb = 2/sigmub
    ## 
    ## obsCLs = CLs(q1sb,q1Asb)  # CLs function assumes q1Asb = -q1Ab
    ## expCLs = CLs(q1Ab,q1Asb)  # median (expected) values of CLs
   
    # hmm I'm a little confused about that... seems to work fine, but value of
    # q should not depend on mu hypothesis. Should just come from likelihood, which
    # is here assumed to be
    # -2*log(L_{s+b}/L_{b})
    #  = -2*[  n*log(s+b) - (s+b) - n!
    #        - n*log(b) + b + n! ]
    #  = -2*[ n*log((s+b)/b) - s ]
    # Might make more sense to work in a different order. Try this:
    # (this way definitely ignores systematics, but I am not sure that they were treated
    # correctly the other way anyhow. Might need to rethink how to include those properly.
    # (i.e. profile or marginalise likelihood I guess)
    # For now just use simulator.cpp for this and do it numerically)

    ## # raw Poisson q (no systematics)
    ## def q0(n): 
    ##    return -2*(n*np.log((s+b)/b) - s)

    # q in Gaussian limit, marginalised over systematics.
    def q(n):
       var_sb = s + (s*fsigs)**2 + sigb**2   
       var_b  = sigb**2
       chi2sb = (n - (s+b))**2/var_sb 
       chi2b  = (n - b)**2/var_b
       return chi2sb - chi2b # + np.log(var_sb/var_b) # last term seems to be causing problems at low s, not totally sure why. Removing it gives excellent match to the raw Poisson formula in absence of systematics...
    # Seems to match simulator.cpp results quite nicely, for large number of counts anyway
    # Same sort of trick should apply to the correlated case

    qobs = q(n)
    qAsb = q(s+b)
    qAb  = q(b)
  
    obsCLs = CLs_2(qobs,qAsb,qAb)
    expCLs = CLs_2(qAb, qAsb,qAb)

    ## expCLsapprox = 2*( 1-rv.cdf(sqrm(q1Asb)) )  #should exactly equal expCLs since 
                              # it simply makes analytic use of q1obs = q1Ab
    
    ## psb = 1 - rv.cdf( (q1sb+1/sigmusb**2)/(2/sigmusb) ) 
    ## pb  =     rv.cdf( (q1b -1/sigmub**2 )/(2/sigmub ) )
    ## obsCLsexact = psb/(1-pb)  # does not assume q1Asb ~ -q1Ab
    #print "obsCLs, obsCLsexact", obsCLs, obsCLsexact i

    ## print 'mu=',mu
    ## print 's=',s
    ## print 'fsigs=',fsigs
    ## print 'sigb=',sigb
    ## print 'sigmusb',sigmusb
    ## print 'sigmub',sigmub
    ## print 'q1Asb',q1Asb
    ## print 'q1Ab',q1Ab
    ## print 'sigqsb',sigqsb
    ## print 'sigqb',sigqb

    if returnextra:
       return obsCLs, expCLs, qobs, qAsb, qAb #, sigqsb, sigqb
    return obsCLs      

# Note: according to the above approximations, the test statistic QCLs should be
# distributed as a Gaussian with mean = (1-2*mu)/sigma**2 and variance = 4/sigma**2
# (arXiv: 1007.1727 eq. 73,74)

def CLs_corr(a1,a2,s1,s2,rho):
   """Computes combined CLs values for two experiments with corellated event counts (ignores systematics!)
      a1 - experiment 1
      a2 - experiment 2
      s1 - predicted signal in a1
      s2 - predicted signal in a2
      rho - linear correlation coefficient
   """
   mu1 = (a1.o - a1.b)/s1
   mu2 = (a2.o - a2.b)/s2  
   sig1 = np.sqrt(1*s1 + a1.b)/s1 # + (1*s*fsigs)**2 + sigb**2)/s  #mu'=1
   sig2 = np.sqrt(1*s2 + a2.b)/s2 # " " 
   
   qobs = qcomb(mu1,sig1,mu2,sig2,rho)
   qAsb = qcomb(1,sig1,1,sig2,rho)
   qAb  = qcomb(0,sig1,0,sig2,rho)

   obsCLs = CLs(qobs,qAsb)  # CLs function assumes q1Asb = -q1Ab
   expCLs = CLs(qAb,qAsb)   # median (expected) values of CLs
   
   qAbp = qcomb( sig1,sig1, sig2,sig2,rho)
   qAbm = qcomb(-sig1,sig1,-sig2,sig2,rho)

   #+/- 1 sigma
   expCLsp = CLs(qAbp,qAsb)
   expCLsm = CLs(qAbm,qAsb)

   return obsCLs,expCLs,expCLsp,expCLsm


#=======================================================================
# DEFINE MODEL AND EXPERIMENT
#=======================================================================

class Experiment:

   # Initialisation   

   def __init__(self,name,o,ssys,b,bsys=0.,bstat=None,sK=1,outdir='',obslim=None,explim=None):
      # Identifier
      self.name = name
      # Directory to store results
      self.outdir = outdir
      # Observed number of events
      self.o = o
      # Gaussian systematic uncertainty on signal yield (fractional)
      self.ssys = ssys
      # Signal efficiency scaling factor
      self.sK = sK
      # Expected number of background counts
      self.b = b
      # Gaussian statistical uncertainty on background yield (from monte carlo, limited control regions, etc.). If none provided, assume Poisson (asymptotic)
      if bstat==None:
         self.bstat = np.sqrt(b)/b #still fractional
      else:
         self.bstat = bstat
      # Gaussian systematic uncertainty on background yield (fractional)
      self.bsys = bsys
      # Total background uncertainty
      self.bsystot = np.sqrt(self.bsys**2 + self.bstat**2)
      # Background +/-1 sigma counts
      self.bp1 = np.round(b + b*self.bsystot)
      self.bm1 = np.round(b - b*self.bsystot)

      self.nlist={'obs':   ('r', self.o  ),
                  'b-1sig':('b', self.bm1),
                  'b' :    ('k', self.b),
                  'b+1sig':('g', self.bp1)}

      # Observed and expected limits reported by actual experiment; used to tune uncertainty parameters to reproduce the correct limits.
      self.obslim = obslim
      self.explim = explim

   # Analysis tools

   def likefunc(self,n,s,dels,delb):
       """log-likelihood function for individual parameter points in the model.
       Contains the two nuisance parameters dels and delb, which
       parameterise the systematic errors. Marginalise these out to be
       Bayesian, or profile them to be pseudo-frequentist (they still
       have priors).
       The parameter 's' (signal mean) should then be the only free 
       parameter left. 
       Args:
       n - observed number of events
       i - which signal region we are currently looking at
       dels - systematic error parameter for signal
       delb - systematic error parameter for background
       s - expected number of events due to signal
       ssys - estimated gaussian uncertainty on expected number of signal events (effectively a prior)
       b - expected number of events due to background
       bsys - estimated gaussian uncertainty on expected number of background events (effectively a prior)
       bstat - estimated "statistical" gaussian uncertainty on expected number of background events (also effectively a prior)
       K - signal efficiency scaling factor
       """
       #bsystot = np.sqrt(self.bsys**2 + self.bstat**2)                             # assume priors are independent
       siglike = logpoissonlike(n,self.sK*s*(1+dels*self.ssys)+self.b*(1+delb*self.bsystot))  # poisson signal + background log likelihood
       #Need to change the scaling of the prior to match the simulated data.
       #Makes no difference to inferences.
       Pdels = pymc.normal_like(dels,0,1) #+ 0.5*np.log(2*np.pi)            #standard normal gaussian log prior on dels
       Pdelb = pymc.normal_like(delb,0,1) #+ 0.5*np.log(2*np.pi)            #standard normal gaussian log prior on delb
       
       if siglike + Pdels + Pdelb < -1e200:
           print dels, delb
           print siglike,Pdels,Pdelb, self.sK*s*(1+dels*self.ssys)+self.b*(1+delb*self.bsystot)
           raise
       
       return siglike + Pdels + Pdelb


   def getprofchi2(self,n,s,getsigmas=False):
       """Find minimum -2*Log(likelihood) value for a given set of observations n
       and s, varying dels and delb (i.e. profiling out nuisance parameters)
       Args:
       see 'likefunc' for argument definitions
       getsigmas - Set to True if 1 and 2 sigma intervals for s are also
       desired.
       
       Note!
       n - number of observed counts in this signal region.
       Must be set globally, so set to desired experimental value before calling this function
       """
       #define function to be minimised
       #global res
       #res = []
       
       def chi2(dels,delb):
           """gets the chi**2 for this signal mean and systematic error
           parameters"""
           global res
           #print 'likefunc', likefunc(n,dels,delb,s,ssys,b,bsys,bstat)
           r = -2*self.likefunc(n,s,dels,delb)
           #res += [[dels,delb,r]]
           return r
           
       mi = minuit.Minuit(chi2, dels=0., err_dels=self.ssys, delb=0., err_delb=np.sqrt(self.bsys**2+self.bstat**2),
               limit_dels=(-1./self.ssys,10), limit_delb=(-1./np.sqrt(self.bsys**2+self.bstat**2),10))
               #need the limits to prevent either of the signal or background means from going negative
       mi.maxcalls = 100000
       mi.strategy = 2
       mi.tol = 10
       try:
           mi.migrad() #run Minuit's MIGRAD optimized gradient-based minimum search
       except minuit.MinuitError, err:
           print "Warning, Minuit encountered a problem: {0}".format(err)
      
       return mi.values['dels'], mi.values['delb'], mi.fval

   
   def getminchi2(self,n,sinit,serr,getsigmas=False):
       """Find minimum -2*Log(likelihood) value for a given set of observations n,
       varying s, dels and delb.
       Args:
       see 'likefunc' for argument definitions
       sinit - initial value to give s parameter
       serr - starting step-size for minimum search
       getsigmas - Set to True if 1 and 2 sigma intervals for s are also
       desired.
       
       Note!
       n - number of observed counts in this signal region.
       Must be set globally, so set to desired experimental value before calling this function
       """
       #define function to be minimised
       def chi2(s,dels,delb):
           """gets the chi**2 for this signal mean and systematic error
           parameters"""
           return -2*self.likefunc(n,s,dels,delb)
   
       mi = minuit.Minuit(chi2, s=sinit, err_s=serr, dels=0., err_dels=self.ssys, delb=0., err_delb=np.sqrt(self.bsys**2+self.bstat**2),
           limit_dels=(-1./self.ssys,10), limit_delb=(-1./np.sqrt(self.bsys**2+self.bstat**2),10), limit_s=(0,1000))
           #need the limits to prevent either of the signal or background means from going negative
       mi.maxcalls = 100000
       mi.strategy = 2
       mi.tol = 100
       try:
           mi.migrad() #run Minuit's MIGRAD optimized gradient-based minimum search
       except minuit.MinuitError, err:
           print "Warning, Minuit encountered a problem: {0}".format(err)
       #print mi.values
       if getsigmas:
           #use Minuit MINOS algorithm to compute xs values that raise chi2 from the minimum by sigma^2
           #The values returned are the ERRORS relative to the best fit value.
           mi.minos('s',-1)
           mi.minos('s',1)
           mi.minos('s',-2)
           mi.minos('s',2)
           return mi.values['s'], mi.values['dels'], mi.values['delb'], mi.fval, mi.merrors #m.merrors contains a dictionary from parameter-number of sigma pairs to the MINOS result
       else:
           return mi.values['s'], mi.values['dels'], mi.values['delb'], mi.fval
           
   def getmargchi2(self,n,s,getsigmas=False):
       """Find marginalised -2*Log(likelihood) value for a given set of observations n
       and s, varying dels and delb (i.e. marginalising out nuisance parameters)
       Args:
       see 'likefunc' for argument definitions
       getsigmas - Set to True if 1 and 2 sigma intervals for s are also
       desired.
       
       Note!
       n - number of observed counts in this signal region.
       Must be set globally, so set to desired experimental value before calling this function
       """
       #define function to be minimised
       #global res
       #res = []
       
       def like(dels,delb):
           """gets the likelihood for this signal mean and systematic error
           parameters"""
           #global res
           #print 'likefunc', likefunc(n,dels,delb,s,ssys,b,bsys,bstat)
           r = np.exp(self.likefunc(n,s,dels,delb))
           #res += [[dels,delb,r]]
           return r
       
       #print 'limits:', max(-5,-1./self.ssys), max(-5,-1./np.sqrt(self.bsys**2+self.bstat**2))
       #dblquad has a somewhat stupid order for the limits...
       marglike = integ.dblquad(like,max(-5,-1./np.sqrt(self.bsys**2+self.bstat**2)),5,lambda x:max(-5,-1./self.ssys),lambda x:5, epsabs=1e-3, epsrel=1e-3)    #do marginalisation   
       #print 'done'
       return -2*np.log(marglike[0])
  
   #Maximum likelihood ratio test statistic
   def QMLR(self,n,s,minchi2):
       """-2*(Maximum log-likelihood ratio test statistic)
       s - signal expectation value hypothesis
       minchi2 - Global minimum value of combchi2 (-2*log[likelihood]) found for varying
       s, dels and delb (signal and background systematics) (list of possible minchi2's corresponding
       to different numbers of observed events)
       
       Note!
       n - list of observed counts in each bin, in bin order (feeds into combchi2 function).
       Must be set globally, so set to desired experimental values before calling this function
       
       Also note!
       This statistic is more positive for minchi2-like signals, and more negative for signals
       more like the hypothesised model. So 'more extreme' means 'more negative'.
       """
       #print s, minchi2
       #need to profile over dels and delb 
       print 'n',n,'s',s, getprofchi2(n,s,ssys,b,bsys,bstat)
       print 'n',n, minchi2[n]
       #return getmargchi2(n,s,ssys,b,bsys,bstat) -  minchi2[n]
       #print n, len(minchi2)
       #r = getprofchi2(n,s,ssys,b,bsys,bstat)[2]
       r =  self.getmargchi2(n,s)
       return r -  minchi2[n], r     #second part of tuple records the fitted parameter values 

   #CLs test statistic
   def QCLs_prof(self,n,s,chi2B=None):
      """
      Compute the value of the test statistic:
                      /  L(n, mu=1, nuis) \
      Q(n) = - 2 * ln | ----------------- |
                      \  L(n, mu=0, nuis) /
      where nuisance parameters are profiled out.
      """
      # the following return
      # (mi.values['dels'], mi.values['delb'], mi.fval),
      # we just want the profiled value mi.fval
      chi2SB= self.getprofchi2(n,s)[2]
      # chi2B only varies with n, so to save time a previously computed value may be supplied
      if chi2B == None:
         chi2B = self.getprofchi2(n,0)[2]
      return (chi2SB-chi2B, chi2B)

   #CLs test statistic (no profiling; this is what is used in simulator.cpp (at time of writing this...))
   def QCLs(self,n,s,chi2B=None):
      """
      Compute the value of the test statistic:
                      /  L(n, mu=1) \
      Q(n) = - 2 * ln | ----------- |
                      \  L(n, mu=0) /
      where nuisance parameters are assumed to adopt their expected values
      """
      
      chi2SB = -2*self.likefunc(n,s,0,0)

      # chi2B only varies with n, so to save time a previously computed value may be supplied
      if chi2B == None:
         chi2B = -2*self.likefunc(n,0,0,0)

      return (chi2SB-chi2B, chi2B)


   def getCL(self,fname,n,svals=None,regen='ifNone',method='marg',N=2000,regenQdist='ifNone',simall=True,savedata=True):
      """Generate CLs or CLsb data

      Determine how CLs related values vary with analysis parameters
      
      Args:
      svals - array of signal mean hypotheses to test
      fname - name of file in which to store (or from which to retrieve) results
      n     - observed number of signal candidates
      regen - whether to regenerate the results or just retrieve them from fname
      method - which method to use to compute the p values
      simall - True or number: auxilliary flag for 'simulate' method. If not True, only simulates CLs test statistic distributions for small signal+background hypotheses, and uses asymptotic approximation otherwise. The change point is the number set in this flag.
      N - number of pseudoexperiments to run (valid for 'simulate' method only)
      savedate - Whether to store results for the future. For tuning limits it is useful to set this to False to avoid erasing old data.
      """
      if method=='asymptotic':
         #-----------------------------------------------------------------------------------
         # Computes CLs values using asympotic approximations for test statistic distribution
         #------------------------------------------------------------------------------------
         CLs = CLs_asymp(n,svals,self.b,self.b*self.bsystot,self.ssys) 
         return (CLs,svals)
 
      elif method=='simulate':
         #------------------------------------------------------------------------------
         # Computes CLs values using simulated distributions for the test statistic
         #------------------------------------------------------------------------------i
         
         def write2pkl(Qdists):
             if savedata==True:
                 with open(self.outdir+"/"+self.name+"-Qdists_N={0}.pkl".format(N),'w') as f:
                     p.dump(Qdists,f)
             else:
                 print "Warning! Not saving results of test statistic distribution simulations!"
 
         def getQdists(svals_v):
             # Get distributions of Q, for given s, from file if we have them, else generate them
             def genQdists(s):        
                 
                 # Perform pseudoexperiments
                 #-----------------

                 # Simulate test statistic assuming S+B hypothesis is true
                 QSBvals,QSBfreq = self.get_QCLs_dist_cpp(s,muT=1,N=N)
                 # Simulate test statistic assuming B hypothesis is true
                 QBvals,QBfreq = self.get_QCLs_dist_cpp(s,muT=0,N=N)
 
                 showfig=False
                 if showfig:
                    Qobs,chi2B = self.QCLs(n,s)
                    psb = sum([pi for qi,pi in zip(QSBvals,QSBfreq) if Qobs<qi])
                    pb  = sum([pi for qi,pi in zip(QBvals, QBfreq)  if qi<=Qobs])
                    fig = plt.figure(figsize=(5,5))
                    ax = fig.add_subplot(111)
                    # ax.stem(QSBvals,QSBfreq,markerfmt='ro',linefmt='r-',alpha=0.7) 
                    # ax.stem(QBvals,QBfreq,markerfmt='bo',linefmt='b-',alpha=0.7)
                    ax.plot(QSBvals,QSBfreq,'ro')
                    ax.vlines(QSBvals,0,QSBfreq,colors='r',alpha=0.5)
                    ax.plot(QBvals,QBfreq,'bo')
                    ax.vlines(QBvals,0,QBfreq,colors='b',alpha=0.5)
                    ax.axvline(x=Qobs,color='orange',lw=2)
                    ax.set_title("s+b={0} (s={1}), n={2}\npsb={3}, pb={4}\nCLs={5}".format(s+self.b,s,n,psb,pb,psb/(1-pb)))
                    ax.set_ylim(bottom=0)
                    plt.tight_layout()
                    plt.show()
 
                 # Stash distributions for later             
                 return ((QSBvals,QSBfreq),(QBvals,QBfreq))

             if regenQdist==False or regenQdist=='ifNone':
                 try:
                     with open(self.outdir+"/"+self.name+"-Qdists_N={0}.pkl".format(N),'r') as f:
                         Qdists = p.load(f)
                 except IOError:
                     if regenQdist=='ifNone':
                         print 'No Qdist file found for this number of trials, generating all Q distributions from scratch...'
                         Qdists = {}
                         for s in svals_v: 
                             Qdists[s] = genQdists(s)
                         write2pkl(Qdists)
                     else:
                         raise             
                     return Qdists
   
                 # check if any of the s values we want are missing
                 missings = [s for s in svals_v if s not in Qdists.keys()]
                 if len(missings)>0:
                     if regenQdist==False:
                         raise ValueError('Pre-generated QCLs distributions values not found for the following s values:\n{0}\nPlease set regenQdist=ifNone (or True) and run again')
                     else: 
                         for s in missings: 
                             Qdists[s] = genQdists(s)
                     write2pkl(Qdists)
   
                 return Qdists
             
             if regenQdist==True:
                 Qdists = {}
                 for s in svals_v:
                     Qdists[s] = genQdists(s)
                 write2pkl(Qdists)
                 return Qdists
                 
         def genCLvals():
             CLs = np.zeros(len(svals))
             if simall!=True:
                tosim = np.where( svals+self.b <= simall ) #simulate for these values
                notsim = np.where( svals+self.b > simall ) #do not simulate for these values
             else:
                # stick inside tuple to match results of np.where
                tosim = (np.arange(len(svals),dtype=np.int),) #simulate everything
                notsim = (np.array([],dtype=np.int),)

             #print "Simulating {0} hypotheses...".format(len(tosim[0]))

             # Simulate for the specified s hypotheses
             Qdists = getQdists(svals[tosim])
             for i,s in zip(tosim[0],svals[tosim]):                   
                 (QSBvals,QSBfreq),(QBvals,QBfreq) = Qdists[s]

                 # Compute CLs
                 #----------------
                 # Compute observed value of test statistic
                 Qobs,chi2B = self.QCLs(n,s)
                 # Compute p_s+b, p_b, and CLs
                 # < and <= chosen to produce conservative CLs value.
                 psb = sum([pi for qi,pi in zip(QSBvals,QSBfreq) if Qobs<qi])
                 pb  = sum([pi for qi,pi in zip(QBvals, QBfreq)  if qi<=Qobs])

                 # Deal with divide by zero situation
                 if pb==1.: CLs[i] = 0.   
                 else: CLs[i] = psb/(1.-pb)

             # Use asymptotic approximations for the rest
             if len(svals[notsim])!=0:
                 CLs[notsim] = CLs_asymp(n,svals[notsim],self.b,self.b*self.bsystot,self.ssys)       
             
             #Write results to file
             if savedata==True:
                 with open(fname+"sim",'w') as f:   #save them in a file identified by the mass hypothesis
                     p.dump((CLs,svals),f) 
             else:
                 print "Warning! Not saving results of CLs curve determination."
           
             return (CLs,svals)
 
         if regen==True:
             (CLs,svals) = genCLvals()
         else:
             #If the CL vals have been pre-generated, load them from the file
             try:
                 with open(fname+"sim",'r') as f:
                    (CLs,svals) = p.load(f)
             except IOError:
                 if regen=='ifNone': 
                    (CLs,svals) = genCLvals()
                 else:
                    print 'Pre-generated CL values not found, please set regen=True and run again'
                    raise
         return (CLs,svals)
        
      elif method=='marg':
         #---------------------------------------------------------
         # Computes CLs+b values in a simplified fashion
         #----------------------------------------------------------
         #Definitions:
         # CLs+b - CL value assuming Q follows the signal+background PDF
         def genCLvals():
             CLsb = np.zeros(len(svals))
             for i,s in enumerate(svals):                    
                 # Following Allanach et al method, chi2 doesn't work for marginalised likelihood
                 likelist = [np.exp(-0.5*self.getmargchi2(ni,s)) for ni in range(int(n)+1)]
                 print 's', s, 'cumu. like', sum(likelist), likelist
                 CLsb[i] = sum(likelist)                         
             #Write all these distributions to file
             if savedata:
                 with open(fname,'w') as f:   #save them in a file identified by the mass hypothesis
                     p.dump((CLsb,svals),f) 
             else:
                 print "Warning! Not saving results of CLs curve determination."
             return (CLsb,svals)
   
         # CLb - CL value assuming Q follows the background-only PDF
         # Integral(Q_PDF) from Qobs to +Inf = CL
         # I.e. the CL is 1 minus the probability of observing a test statistic value this large (in negative direction) or
         # larger if the test statistic follows the assumed distribution.
       
         if regen==True: 
             (CLsb,svals) = genCLvals()
         else:
             #If the CL vals have been pre-generated, load them from the file
             try:
                 with open(fname,'r') as f:
                    (CLsb,svals) = p.load(f)
             except IOError:
                 if regen=='ifNone': 
                    (CLsb,svals) = genCLvals()
                 else:
                    print 'Pre-generated CL values not found, please set regen=True and run again'
                    raise
   
         return (CLsb,svals)

      else:      
         raise ValueError("Invalid method for computing p-values specified!")
 

   # Plotting/analysis related routines

   def getCLvals(self,svals,regen,method='simulate',N=2000,regenQdist='ifNone',savedata=True,simall=True):
      self.CL = {}
      self.CLfunc = {}
      for i,(label,(color,nobs)) in enumerate(self.nlist.items()):
         if (i>0) and (savedata==True): regenQdist=False #should only have to do this once (if data saved...)
         CLvals, svals = self.getCL(fname='{0}/{1}-{2}.pkl'.format(self.outdir,self.name,label),
                                      n=nobs,svals=svals,regen=regen,method=method,N=N,
                                      regenQdist=regenQdist,savedata=savedata, simall=simall)
         self.CL[label] = (svals, CLvals)
         # Create interpolating functions for each observation
         #self.CLfunc[label] = extrap1d(interp.interp1d(svals,CLvals,bounds_error=False,fill_value=0))
         # no need for extrap I think
         self.CLfunc[label] = interp.interp1d(svals,CLvals,bounds_error=False,fill_value=0)

   # As above, but does it just for a single sval, with no interpolation or saving of results
   def getCL_direct(self,s,method='simulate',N=2000,nobs=None,simall=True):
      if nobs==None: nobs=self.o
      CLvals, svals = self.getCL(fname='{0}/{1}-{2}.pkl'.format(self.outdir,self.name,'tmp'),
                       n=nobs,svals=np.atleast_1d(s),regen=True,method=method,N=N,
                       regenQdist=True,savedata=False,simall=simall)
      return CLvals
       
   def plotCL(self,ax=None,scale=1):
      """Plot curves of CL values"""
      if ax==None:
         fig1 = plt.figure(figsize=(6,4))
         #plt.subplots_adjust(left=0.08,right=0.97,bottom=0.07,top=0.93,hspace=0.3)
         axCL = fig1.add_subplot(111)
      else:
         axCL = ax
      for label,(color,nobs) in self.nlist.items():
         svals, yCL = self.CL[label]
         axCL.plot(svals*scale,yCL,ls='-',color=color,label=r'$CL$ '+label,alpha=0.7)
      axCL.plot(svals*scale,[0.05]*len(svals),'k--',label=r'$CL=0.05$')
      axCL.set_title('CL values for {0}'.format(self.name))
      axCL.set_xlabel('s')
      axCL.set_ylabel('CL')
      axCL.set_ylim((0,1))
         
      leg = axCL.legend() #second argument is labels
      leg.get_frame().set_alpha(0.5)
      if ax==None:
         fig1.tight_layout()
         fig1.savefig('{0}/{1}_CL.png'.format(self.outdir,self.name))

   def interpolate_pvalues(self,label,sarray):
      """Use the CL values calculated for various s hypotheses to interpolate p values onto an array of s values."""
      return self.CLfunc[label](sarray)

   def exp_excluded(self,sarray):
      """Return True if s > self.explim"""
      if self.explim==None:
         raise ValueError("Cannot compare s to explim! No explim has been supplied!")
      return sarray > self.explim
 
   def obs_excluded(self,sarray):
      """Return True if s > self.obslim"""
      if self.obslim==None:
         raise ValueError("Cannot compare s to obslim! No obslim has been supplied!")
      return sarray > self.obslim
 
   def plot2Dlimit(self,ax,coords,slist,(gridx,gridy),colormap=None):
      """Plot limits resulting from interpolation of 1D list of s hypotheses onto a regular grid
      Args:
      ax - axis object on which to place plot
      coords - N*2 array of coordinates corresponding to s hypotheses
      slist -  N*1 array/list of s hypotheses
      gridx, gridy - 1D lists of grid x and y values from which to build target grid
      """
      out={}
      for label,(color,nobs) in self.nlist.items():
         #print label, nobs
         pvals = self.interpolate_pvalues(label,slist)
         #for s,p in zip(slist,pvals):
         #   print s,p
         grid_x, grid_y = np.meshgrid(gridx,gridy)
         gridpvals = interp.griddata(coords,pvals,(grid_x,grid_y),method='linear')
         #gridpvals = grid2duneven(coords,pvals,(gridx,gridy),unevenaxis=1,tol=0)
         
         if colormap==label:
            im = ax.imshow(gridpvals, interpolation='bilinear', origin='lower',
                      extent=(min(gridx),max(gridx),min(gridy),max(gridy)), aspect='auto',alpha=0.5)
            cbar = plt.colorbar(im, ticks=[0, 0.05, 0.1, 0.32, 1])
   
         CS = ax.contour(grid_x, grid_y, gridpvals, levels=[0.05], linewidths=2, colors=[color])   

         out[label]=gridpvals
      
      # return grid of pvalues; useful for other plots
      return out

   #===========================================================================
   # PyMC tools for generating distributions of test statistics via Monte Carlo
   #=========================================================================== 

   def create_model(self,s,muT):
      """Create PyMC stochastic model for the experiment

      s - signal event rate hypothesis
      muT - true value of signal strength parameter (zero or 1)
      """
      b = self.b
      sigmas = s*self.ssys
      sigmab = self.b*self.bsystot

      # Define the model class
      class likelihood_model: 
          
         # Stochastic variables for signal, background, and total event rates
         #signal_rate     = pymc.Normal('signal_rate',     mu=s*muT,  tau=1/sigmas**2)
         #background_rate = pymc.Normal('background_rate', mu=b,      tau=1/sigmab**2)
         # Doh, need to use truncated normal to prevent negative values
         signal_rate     = pymc.TruncatedNormal('signal_rate',     mu=s*muT, tau=1/sigmas**2, a=0, b=np.inf)
         background_rate = pymc.TruncatedNormal('background_rate', mu=b,     tau=1/sigmab**2, a=0, b=np.inf)
        
         # Deterministic variable (simply the sum of the signal and background rates)
         total_rate = pymc.LinearCombination('total_rate', [1,1], [signal_rate, background_rate])
         # Stochastic variable for number of observed events
         observed_events = pymc.Poisson('observed_events', mu=total_rate)
        
         # Deterministic variable for the test statistic
         @pymc.deterministic()
         def qCLs(n=observed_events):
            q,chi2B = self.QCLs(n,s) 
            return q

      return likelihood_model

   def get_QCLs_dist_PyMC(self,s,muT,N=20000):
      """Determine the distribution of the CLs test statistic for the supplied signal hypothesis
      (using with muT=1 or muT=0 to set the S+B or B hypotheses as true respectively).
      
      This version of get_QCLs_dist uses PyMC for the test statistic generation      

      Note that it is a python function being evaluated in the loop, so the mcmc is pretty slow.
      """
      print "Determining qCLs(muT={0}) distribution for s={1}...   ".format(muT,s)

      # Generate the stochastic model
      model = self.create_model(s,muT)
 
      # Run the mcmc
      mcmc = pymc.MCMC(model)
      mcmc.sample(N,0,1) #don't need burn in or thinning, just getting random samples.
      Qsamples = mcmc.trace("qCLs")[:]  

      # Possible test statistic values are discrete, so count up occurances of each   
      keys, counts = count_unique(Qsamples)
      normalised_counts = counts/len(Qsamples)
    
      return (keys,normalised_counts)      

   def get_QCLs_dist_cpp(self,s,muT,N=20000):
      """Determine the distribution of the CLs test statistic for the supplied signal hypothesis
      (using with muT=1 or muT=0 to set the S+B or B hypotheses as true respectively).            

      This version of get_QCLs_dist uses my custom written c++ module to compute the
      test statistic values. Should be much faster than PyMC. 
      """
      print "Determining qCLs(muT={0}) distribution for s={1}...   ".format(muT,s)

      # Note:
      # self.sys is the SYSTEMATIC uncertainty on the signal RATE. No `natural' Poisson stuff included
      # self.bsystot is the uncertainty on the TOTAL YIELD due to the background. Includes the standard Poisson error PLUS other systematics.
      # The simulator necessarily separates the Poisson error from the error on the rate. So we have to
      # make sure this happens first. The user is permitted to change the statistical error from the
      # Poisson amount, so I am here going to move all of this into the systematic parameter, and let
      # the simulator deal with the Poisson statistics itself.
      # ( note  (np.sqrt(b)/b)**2 = 1/b )
      ssys = self.ssys
      bsys = np.sqrt(self.bsystot**2 - 1/self.b)
      if bsys<0 or np.isnan(bsys): 
        #print self.bsystot
        #print self.bsys
        #print self.bstat
        #print bsys
        #print "Warning, negative background systematic calculated, please check code and inputs... (often just due to rounding error... setting systematic to zero)"
        bsys=0.

      Qsamples = sim.simulate(muT,s,self.b,ssys,bsys,self.sK,N)

      # Possible test statistic values are discrete, so count up occurances of each   
      keys, counts = count_unique(Qsamples)
      normalised_counts = counts/len(Qsamples)
    
      return (keys,normalised_counts)     

   def autotune(self,fname,method='simulate'):
      """Algorithm for automatically tuning systematic uncertainty parameters to optimise fit of 
      the model likelihood to the observed and expected limits reported by the experiments"""
     
      # Parameters to be optimised:
      # Gaussian systematic uncertainty on signal yield (fractional)
      #self.ssys = ssys
      # Signal efficiency scaling factor
      #self.sK = sK
    
      # Record original parameter values so we can be them back how we found them when we are finished.
      orig_ssys = self.ssys
      orig_sK = self.sK
      orig_o = self.o
      #define function to be minimised
      def pseudochi2(ssys,sK):
          """pseudo-chi2 for optimising systematic parameters"""
          self.ssys = ssys
          self.sK = sK
 
          # Compute CLs value at the signal value that is supposed to be the observed and expected and limit
          # Should be as close to 0.05 
          expCL = self.getCL_direct(s=self.explim,method='simulate',N=100000,nobs=self.b,simall=True)
          obsCL = self.getCL_direct(s=self.obslim,method='simulate',N=100000,nobs=self.o,simall=True)
          print 'ssys = {0}; sK {1}'.format(ssys,sK)
          print 'explim CL: {0}, (target={1})'.format(expCL, 0.05)
          print 'obslim CL: {0}, (target={1})'.format(obsCL, 0.05)

          return (expCL - 0.05)**2/0.01**2 + (obsCL - 0.05)**2/0.01**2
  
      mi = minuit.Minuit(pseudochi2, ssys=self.ssys, err_ssys=0.1, sK=self.sK, err_sK=0.1)
      print mi.errors

      mi.maxcalls = 100
      mi.strategy = 2
      mi.tol = 1
      mi.up = 1
      try:
          mi.simplex() #run Minuit optimisation (migrid seems to ignore error parameters so using simplex)
      except minuit.MinuitError, err:
          print "Warning, Minuit encountered a problem: {0}".format(err)
          print mi.values
     
      f = open(fname,"a") 
      f.write("Optimised values for {0} are: \n".format(self.name))
      f.write("  ssys = {0}\n".format(mi.values['ssys']))
      f.write("  sK   = {0}\n".format(mi.values['sK']))
      f.write("  (pseudo-chi^2 = {0})\n".format(mi.fval))
      f.close()

      self.ssys = mi.values['ssys'] 
      self.sK = mi.values['sK']
  
      return
