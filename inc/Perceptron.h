#pragma once
#include "globals.h"
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <functional>

class Perceptron
{
public:
    // Constructors
    Perceptron();
    Perceptron(std::vector<int> geometry, bool valid);
    Perceptron operator=(const Perceptron &per);
    ~Perceptron(){};
    
    // Feed-forward evaluation
    double evaluate_at(Event*, int);
    
    // GA functions
    void perturb(int gens);
    
    // BP functions
    void backprop(double alpha);
    double delta(int l, int n);
    
    // Saving/discarding configurations
    void revert() {_currentWeights = _bestWeights; _outputs = _prevoutputs;}
    void stash() { _bestWeights = _currentWeights; _prevoutputs = _outputs;}
    void stash_gen(int gen) {_best_generation = gen;}

    // Get/Access functions
    bool isValidation() {return _isValidation;}
    weightNet currentWeights() {return _currentWeights;}
    weightNet bestWeights() {return _bestWeights;}
    outputNet outputs() {return _outputs;}
    std::vector<int> geometry() {return _geometry;}
    void copy_weights(const Perceptron &per){_currentWeights = per._currentWeights;}
    void copy_weights(weightNet weightnet) {_currentWeights = weightnet;}
    void add_to_weight(int l, int n, int w, double incr) {_currentWeights[l][n][w] += incr;}
    int best_gen() {return _best_generation;}
    
    
private:
    std::vector<int> _geometry;     // Layout of NN
    bool _isValidation;             // If the current input is from validation sample
    
    weightNet _currentWeights;
    weightNet _bestWeights;
    
    int _best_generation;
    
    outputNet _outputs;
    outputNet _prevoutputs;
    bool _truth;
};

