#include "Perceptron.h"


double sigmoid(double x){
    return 1.0/(1.0+exp(-1.0*x));
    }
    
Perceptron::Perceptron(){}

Perceptron::Perceptron(std::vector<int> geometry, bool valid): _geometry(geometry), _isValidation(valid){
    // geometry: list of number of nodes in each layer including input - use to fill in random weights
    
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<double> gauss(0, 1.0);
    
    // Placeholder for input layer
    std::vector< Event > dummy_weights = {std::vector<double>(1)};
    _currentWeights.push_back(dummy_weights); 
    
    // Filling hidden layers with random weights
    for (int layer = 1; layer < _geometry.size(); layer++){ 
        
        std::vector<std::vector<double> > thislayer = {};
        
        for (int nodes = 0; nodes < _geometry[layer]; nodes++){
            
            std::vector<double> node_weights = {};
            
            for (int prev = 0; prev < _geometry[layer-1] + 1; prev++){
                // One for each on the previous layer + a threshold value
                node_weights.push_back(gauss(generator)); 
            }
            thislayer.push_back(node_weights);
        }
        _currentWeights.push_back(thislayer);
    }
    _bestWeights = _currentWeights;
}

Perceptron Perceptron::operator=(const Perceptron &per){
    // Copy constructor
    _currentWeights = per._currentWeights;
    _bestWeights = per._bestWeights;
    _geometry = per._geometry;
    _outputs = per._outputs;
    _isValidation = per._isValidation;
    
    return *this;
}


double Perceptron::evaluate_at(Event* thisEvent, int truth){
    // Evaluatates NN output for some particular event with current weights 
    
    // Clear previous data
    _outputs.clear();
    
    // This event
    _outputs.push_back(*thisEvent);
    _truth = truth;
    
    for (int layer = 1; layer < _currentWeights.size(); layer++){
        std::vector<double> thislayer = {};
    
        for (int node = 0; node < _currentWeights[layer].size(); node++){
            double nodesum = 0;
            
            for (int w = 0; w < _currentWeights[layer][node].size() - 1; w++){
                nodesum += _outputs[layer - 1][w] * _currentWeights[layer][node][w];
            }
            double threshold = _currentWeights[layer][node].back(); //threshold
            nodesum += threshold;
            
            thislayer.push_back(sigmoid(nodesum));
        }
        _outputs.push_back(thislayer);
    }
    
    // FINAL RETURN VALUE
    return _outputs.back()[0];
}

void Perceptron::perturb(int gens){
    // Randomly picks one weight and alters by random additive amount (GA)
    
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<double> gaussian(0, 10.0/sqrt(gens + 1));
    std::uniform_real_distribution<double> uniform(0.0,1.0);
    
    int layer = round((_currentWeights.size()-1.0) * uniform(generator));
    int node = round((_currentWeights[layer].size() - 1.0) * uniform(generator));
    int weight = round((_currentWeights[layer][node].size() -2.0) * uniform(generator))  + 1;
    
    _currentWeights[layer][node][weight] += gaussian(generator);
}


//---------------------------------------------------------------------------------------------------//
//                                           BACKPROP                                                //
//---------------------------------------------------------------------------------------------------//

double Perceptron::delta(int l, int n){
    double thisOut = _outputs[l][n];
    if (l == _currentWeights.size() - 1){
        double delt = thisOut * (1.0 - thisOut) * -1.0 * (_truth/thisOut - (1-_truth)/(1-thisOut));
        return delt;
        }
    else{
        double sum = 0;
        for (int m = 0; m < _currentWeights[l+1].size(); m++){
            sum += delta(l + 1, m) * _currentWeights[l+1][m][n];
            }
        return sum * thisOut * (1.0 - thisOut);
    }
}


void Perceptron::backprop(double alpha){
    for (int l = _currentWeights.size() - 1; l > 0; l--){ // From final layer to 1st hidden layer
        
        for (int n = 0; n < _currentWeights[l].size(); n++){ // All nodes on layer
        
            for (int w = 0; w < _currentWeights[l][n].size(); w++){ 
                // All weights for node <=> 1 for each node on prev layer + threshold
                if (w == _currentWeights[l][n].size() - 1) _currentWeights[l][n][w] += -1.0 * delta(l, n) * alpha;
                
                else _currentWeights[l][n][w] += -1.0 * _outputs[l-1][w] * delta(l, n) * alpha;
            }
        }
    }
}

