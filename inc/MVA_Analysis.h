#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <complex>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/info_parser.hpp>

#include "globals.h"
#include "Perceptron.h"

// Alternative covariance matrix calculation
//#include <gsl/gsl_math.h>
//#include <gsl/gsl_eigen.h>


typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXd;


class MVA_Analysis
{
public:
    MVA_Analysis(std::string card, std::string infile, std::vector<int> geometry);
    MVA_Analysis(std::string card, std::string infile, std::string weightcard);
    ~MVA_Analysis(){};
    
    // I/O
    void readCard(std::string filenm);
    void load_events();
    void save_data();
    void save_weights();
    
    // Preprocessing
    void normalise_inputs();
    void build_Walker_tables();
    void calculate_decoMatrix();
    void decorrelate_inputs(bool);
    
    // Training
    void Train(int gens);
    void Train_GA(int gens);
    void Train_BackProp(int gens);
    double Error(Perceptron&, bool);
    int WeightedSample();
    
    
    // Testing
//  void save_decoInputs();
//  std::vector<double> BP_test(int, int, int, double);
    
private:
    
    // Parameters
    bool _decorrelate;
    int _strategy;
    int _moving_avg_step_size;
    double _max_ratio;
    double _sigwt;
    double _bkgwt;
    double _valTrigger;
    int _BPgens;
    int _GAgens;
    int _sweepsize;
    
    
    // ANNs
    Perceptron _TANN;                               // Training
    Perceptron _VANN;                               // Validation
    double _alpha;                                  // Learning rate (for BP)
    int _elapsedgens;                               // Number of epochs (BP and GA) passed

    // Input
    std::string _infile, _path;
    std::vector<Event> _rawTevents, _rawVevents;    // Pure input
    std::vector<Event> _Tevents, _Vevents;          // After decorrelation
    std::vector<std::string> _Tlabels, _Vlabels;    // Labels for bookkeeping
    std::vector<int> _Ttruth, _Vtruth;              // Truth values (1 or 0)
    std::vector<double> _Tweight, _Vweight;         // Weight associated with
    // NOTE: Order is important! Indices match between these vectors
    // i.e. Vevent 5 will be in _Vevents[4] with its label at _Vlabel[4] etc
    // Shouldn't need to edit any of these except when normalising
    
    // PCA
    MatrixXd _decoMatrix;                           // Decorrelation matrix
    MatrixXd _lambdasqrt;                           // Scaling matrix

    // Sampling & weighting
    std::pair< std::vector<double>, std::vector<int> > _WalkerTable;
    double _sigScale, _bkgScale;
    
    // Errors
    double  _currentVerr, _currentTerr, _stashedVerr, _stashedTerr;
    
    // Output
    std::vector<double> _Terrs, _Verrs, _alphas;
    std::vector< std::vector<double> > _Tresults, _Vresults;
    
    
};

