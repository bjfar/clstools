//
//  Pseudo-experiment generator for clstools.py
//
//  I previously was doing this with pymc, but it was
//  too slow.
//  The interface with Minuit is based on the example 
//  "DemoGaussSim.cpp" in tests/MnSim (in Minuit-1_7_9
//  source tree)
//
//  The general method followed is kind of like the one described here: 
//  http://www.pp.rhul.ac.uk/~cowan/stat/notes/MargVsProf.pdf
//  Here I follow the profile likelihood method, where the background is
//  constrained by a control measurement.
//  ... however we don't actually have a control measurement (or we don't
//  know those details) so we instead treat the signal and background
//  systematics as describing the standard deviation of the distribution
//  of 'error' random variables, whose likelihoods enter the full assessment.
//  This means the distribution is not the same as we use asymptotic formulae for,
//  unfortunately.
//
//  Edit: Hmm, I am not sure the profiling is necessary. The reference above seems
//  to suggest we can evaluate a test statistic based on the expected signal and
//  background instead. This could speed things up a lot.
//
//  Edit 20 Apr 2014: Fixed a couple of bugs in the wrapper code. Now indeed this
//  simulation DOES match the asymptotic formula in the limit where systematics
//  are zero.

// minuit headers
#include "Minuit/FCNBase.h"
#include "Minuit/FunctionMinimum.h"
#include "Minuit/MnMigrad.h"
#include "Minuit/MnUserParameterState.h"

// other headers
#include <Python.h>
#include <iostream>
#include <random>
#include <cmath>
#include <vector>

// Function to be minimised by Minuit (chi2)
// We are profiling out the nuisance parameters dels and delb, so these are the only
// parameters that vary during the minimisation.
class PoissonFcn : public FCNBase 
{
   public:
      // constructor
      PoissonFcn(const double Is, const double Issys, const double Ib, const double Ibsys, const int Imu)
        :  s(Is), ssys(Issys), b(Ib), bsys(Ibsys), mu(Imu) { }

      // destructor
      ~PoissonFcn() { }

      // main function called by Minuit and returning the -2*logl value
      virtual double operator()(const std::vector<double>& pars) const
      {
         double l;
         double dels;
         double delb;

         if( mu==0 )
         {
            delb = pars[0];
            dels = 0.;
         }
         else
         {
            delb = pars[0];
            dels = pars[1];
         }
         // Poisson -2*loglikelihood piece (neglecting constant pieces)
         // summed with
         // Normal -2*loglikelihood pieces (neglecting constant pieces)
         // (note dels and delb have been scaled such that they are standard normal variables)
         l = mu*s*(1+dels*ssys) + b*(1+delb*bsys);
         return -2*( n*std::log(l) - l ) + dels*dels + delb*delb;
      }

      // Chi2 statistic based on expected signal and background
      double chi2exp(int nv) const
      {
         double l;
         // Poisson -2*loglikelihood piece (neglecting constant pieces)
         l = mu*s + b;
         return -2*( nv*std::log(l) - l );
      }

      // set n to a new value
      void set_n(int new_n) { n = new_n; }

      // Errors are defined by the parameter values required to raise the value of the function from the minimum by the amount "up"
      // 1 gets 68% coverage for 1 parameter chi2 fits. To get strict 68% confidence level coverage with 2 free parameters it is a slightly different value, but I don't care about the errors right now so I am not bothering to look up what the correct value is.
      double up() const { return 1.; }      

   private:
      // variables fixed during the minimisation
      int n = 0;
      double s;
      double ssys;
      double b;
      double bsys;
      int mu; //signal strength scaling parameter
};


// QCLs calculation
double QCLs(PoissonFcn& fcnSB, PoissonFcn& fcnB, MnUserParameters uparSB, MnUserParameters uparB)
{
   //----- Get minimum chi2 for S+B hypothesis ----
   
   // Create minimiser and run
   MnMigrad migradSB(fcnSB, uparSB);
   FunctionMinimum min_chi2SB = migradSB();

   //----- Do it again for the B hypothesis -------
   MnMigrad migradB(fcnB, uparB);
   FunctionMinimum min_chi2B = migradB();

   // Compute CLs value (minchi2 for signal hypothesis minus minchi2 for background hypothesis)
   //std::cout<<min_chi2SB.userParameters().params()[0]<<std::endl;
   //std::cout<<min_chi2B.userParameters().params()[0]<<std::endl;
   return min_chi2SB.fval() - min_chi2B.fval();
}

// Main pseudo-experiment "run" function
static PyObject *
simulator_simulate(PyObject *self, PyObject *args)
{
    double sK;     // Signal efficiency scaling factor
    double muT;    // True signal strength parameter value (muT=1 full signal;muT=0 background only)
    double s;      // Hypothesised signal rate mean
    double b;      // Background rate mean
    double sigmas; // Signal rate fractional uncertainty
    double sigmab; // Background rate fractional uncertainty
    long int Ntrials; // Number of pseudo-experiments to run  

    double Q;

    // I think this checks for valid input arguments.
    if (!PyArg_ParseTuple(args, "ddddddl",&muT,&s,&b,&sigmas,&sigmab,&sK,&Ntrials))
        return NULL;

    // Declare python objects 
    PyObject *Qsamples;
    PyObject *item;

    std::random_device rd;
    std::default_random_engine g( rd() ); // Seed random number generator

    std::normal_distribution<double> s_rate(s*sK,s*sK*sigmas);
    std::normal_distribution<double> b_rate(b,b*sigmab);

    Qsamples = PyTuple_New(Ntrials);

    // create function objects to be minimised
    PoissonFcn fcnSB(s*sK, sigmas, b, sigmab, 1);
    PoissonFcn fcnB (s*sK, sigmas, b, sigmab, 0);

    // create Minuit parameters, with start values and step sizes
    //MnUserParameters uparSB;
    //uparSB.add("delb", 0., 1.);
    //uparSB.add("dels", 0., 1.);

    //MnUserParameters uparB;
    //uparB.add("delb", 0., 1.);
 
    // Set limits to prevent negative signal or background rates
    //uparSB.setLimits("delb", (-1./sigmab), 10.);
    //uparSB.setLimits("dels", (-1./sigmas), 10.);

    //uparB.setLimits("delb", (-1./sigmab), 10.);

    for (int i=0; i<Ntrials; ++i)
    {
        // Generate Poisson event with randomly generated rate
        std::poisson_distribution<int> event_count(muT*s_rate(g)+b_rate(g));
        // TESTING! Generate Poisson events with FIXED systematic parameters
        //std::poisson_distribution<int> event_count(muT*s*1.1+b*0.9);
        int number = event_count(g);

        // Calculate QCLs value
        //fcnSB.set_n(number);
        //fcnB.set_n(number);
        //Q = QCLs(fcnSB,fcnB,uparSB,uparB);
        
        // QCLs test statistic based on expected signal and background only (no profiling)
        Q = fcnSB.chi2exp(number) - fcnB.chi2exp(number);

        // Put it into a python tuple for return to python
        item = PyFloat_FromDouble(Q);
        PyTuple_SetItem(Qsamples, i, item);
    }

    return Qsamples;
}

// Method table
static PyMethodDef SimulatorMethods[] = {
    {"simulate",  simulator_simulate, METH_VARARGS,
     "Run pseudoexperiments."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

// Module initialisation function
PyMODINIT_FUNC
initsimulator(void)
{
    (void) Py_InitModule("simulator", SimulatorMethods);
}


