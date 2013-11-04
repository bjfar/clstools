//
//  Pseudo-experiment generator for clstools.py
//
//  I previously was doing this with pymc, but it was
//  too slow.
//  The interface with Minuit is based on the example 
//  "DemoGaussSim.cpp" in tests/MnSim (in Minuit-1_7_9
//  source tree)

// minuit headers
#include "Minuit/FCNBase.h"
#include "Minuit/FunctionMinimum.h"
#include "Minuit/MnMigrad.h"
#include "Minuit/MnUserParameterState.h"

// other headers
#include <Python.h>
#include <iostream>
#include <random>
#include <math.h>
#include <vector>

// Function to be minimised by Minuit (chi2)
// We are profiling out the nuisance parameters dels and delb, so these are the only
// parameters that vary during the minimisation.
class PoissonFcn : public FCNBase 
{
   public:
      // constructor
      PoissonFcn(const double IsK, const double Is, const double Issys, const double Ib, const double Ibsys)
        : sK(IsK), s(Is), ssys(Issys), b(Ib), bsys(Ibsys) { }

      // destructor
      ~PoissonFcn() { }

      // main function called by Minuit and returning the -2*logl value
      virtual double operator()(const std::vector<double>& pars) const
      {
         double l;
         double dels;
         double delb;

         dels = pars[0];
         delb = pars[1];

         // Poisson -2*loglikelihood piece (neglecting constant pieces)
         // summed with
         // Normal -2*loglikelihood pieces (neglecting constant pieces)
         // (note dels and delb have been scaled such that they are standard normal variables)
         l = sK*s*(1+dels*ssys) + b*(1+delb*bsys);
         return -2*( n*std::log(l) - l ) + dels*dels + delb*delb;
      }

      // set n to a new value
      void set_n(int new_n) { n = new_n; }

      // Errors are defined by the parameter values required to raise the value of the function from the minimum by the amount "up"
      // 1 gets 68% coverage for 1 parameter chi2 fits. To get strict 68% confidence level coverage with 2 free parameters it is a slightly different value, but I don't care about the errors right now so I am not bothering to look up what the correct value is.
      double up() const { return 1.; }      

   private:
      // variables fixed during the minimisation
      int n = 0;
      double sK;
      double s;
      double ssys;
      double b;
      double bsys;
};


// QCLs calculation
double QCLs(PoissonFcn& fcnSB, PoissonFcn& fcnB, MnUserParameters upar)
{
   //----- Get minimum chi2 for S+B hypothesis ----
   
   // Create minimiser and run
   MnMigrad migradSB(fcnSB, upar);
   FunctionMinimum min_chi2SB = migradSB();

   //----- Do it again for the B hypothesis -------
   MnMigrad migradB(fcnB, upar);
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
    double sK = 1.;// not using this for now
    double muT;    // True signal strength parameter value (muT=1 full signal;muT=0 background only)
    double s;      // Hypothesised signal rate mean
    double b;      // Background rate mean
    double sigmas; // Signal rate uncertainty
    double sigmab; // Background rate uncertainty
    long int Ntrials; // Number of pseudo-experiments to run  

    double Q;

    // I think this checks for valid input arguments.
    if (!PyArg_ParseTuple(args, "dddddl", &muT,&s,&b,&sigmas,&sigmab,&Ntrials))
        return NULL;

    // Declare python objects 
    PyObject *Qsamples;
    PyObject *item;

    std::random_device rd;
    std::default_random_engine g( rd() ); // Seed random number generator

    std::normal_distribution<double> s_rate(s,sigmas);
    std::normal_distribution<double> b_rate(b,sigmab);

    Qsamples = PyTuple_New(Ntrials);

    // create function objects to be minimised
    PoissonFcn fcnSB(sK, s, sigmas, b, sigmab);
    PoissonFcn fcnB(sK, 0., sigmas, b, sigmab);

    // create Minuit parameters, with start values and step sizes
    MnUserParameters upar;
    upar.add("dels", 0., 1.);
    upar.add("delb", 0., 1.);

    // Set limits to prevent negative signal or background rates
    upar.setLimits("dels", (-1./sigmas), 10.);
    upar.setLimits("delb", (-1./sigmab), 10.);

    for (int i=0; i<Ntrials; ++i)
    {
        // Generate Poisson event with randomly generated rate
        std::poisson_distribution<int> event_count(muT*s_rate(g)+b_rate(g));
        int number = event_count(g);

        // Calculate QCLs value
        fcnSB.set_n(number);
        fcnB.set_n(number);
        Q = QCLs(fcnSB,fcnB,upar);
        
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


