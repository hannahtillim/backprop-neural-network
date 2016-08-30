// TO DO:
// read param card
// write weight card
// read weight card


#include "MVA_Analysis.h"



std::vector<std::string> myExceptions = {"Unable to open file"};

namespace pt = boost::property_tree;

bool debug = false;


//////////////////////////////////////////////////////////////////////////////////////////////////////////
//-------------------------------------- EXTERNAL FUNCTIONS --------------------------------------------//
//////////////////////////////////////////////////////////////////////////////////////////////////////////

double CrossEntropy(double truth, double opinion){
    return -1.0 *(truth * log(opinion) + (1 - truth) * log(1 - opinion));
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//------------------------------------------ CONSTRUCTOR -----------------------------------------------//
//////////////////////////////////////////////////////////////////////////////////////////////////////////

MVA_Analysis::MVA_Analysis(std::string cardfile, std::string infile, std::vector<int> geometry): _infile(infile), _stashedTerr(INFINITY), _stashedVerr(INFINITY) {
    
    readCard(cardfile);
    
    _elapsedgens = 0;
    
    std::vector<Event> temp = {};
    
    // **************************** BOOKING DATA OUTPUT FILES ****************************** //
    
    // Getting the filepath
    std::string filepath = _infile;
    char* slash = "/";
    for (std::string::iterator it = filepath.end(); it != filepath.begin(); it--){
        if (*it != *slash){
            filepath.erase(it, filepath.end());
            }
        else break;
        }
    _path = filepath;
    
    std::ofstream file;
    file.open(_path + "Terrs.txt");
    file.close();
    std::ofstream file2;
    file2.open(_path + "Verrs.txt");
    file2.close();
    std::ofstream file3;
    file3.open(_path + "Results.dat");
    file3.close();
    std::ofstream file4;
    file.open(_path + "NN_Record.dat");
    file.close();
/*    std::ofstream file4;
    file4.open(_path + "Alphas.dat");
    file4.close();
    std::ofstream file5;
    file5.open(_path + "wEvents.dat");
    file5.close();*/
    
    
    // *************************************** INPUT *************************************** //
    
    std::cout << "LOADING EVENTS..." << std::endl;
    
    load_events();
    
    std::cout << "Events loaded" << std::endl;
    
    // *********************************** PREPROCESSING *********************************** //
    
    // Decorrelating
    if (_decorrelate){
        calculate_decoMatrix();
        decorrelate_inputs(false);
        decorrelate_inputs(true);
//        save_decoInputs(); // To check if they look decorrelated
    }
    else{
        _Tevents = _rawTevents;
        _Vevents = _rawVevents;
        }
    
    // Normalising
    normalise_inputs();
    
    // *************************************** ANNS *************************************** //
    
    int numInputs = _Tevents[0].size();
    std::vector<int> geom = {numInputs};
    for (int i = 0; i < geometry.size(); i++){
        geom.push_back(geometry[i]);
    }
    
    std::cout << "INITIALISING ANNs..." << std::endl
    << "\nGeometry: \n" << std::endl
    << "\tInputs: \t\t" << geom[0] << std::endl;
    for (int i = 1; i < geom.size() - 1; i++){
        std::cout << "\tHidden layer " << i << ": \t" << geom[i] << std::endl;
    }
    
    
    _TANN = Perceptron(geom, false);
    _VANN = Perceptron(geom, true);
    _VANN.copy_weights(_TANN);
    
    
    // For random sampling
    build_Walker_tables();
    
    _stashedTerr = INFINITY;
    _stashedVerr = INFINITY;
    
    std::cout << "Neural nets initialised." << std::endl;
    
    
    // *********************************************************************************** //
}

/*
std::vector<double> MVA_Analysis::BP_test(int l, int n, int w, double DW){
    // Tests whether the finite Delta(Error) / Delta(weight) converges to 
    // BP's calculated derivative for small Delta(weight)
    
    std::vector<double> res = {};
    
    std::cout << "weight W_" << l << n << w << std::endl;
    
    // E(w1)
    double op1 = _TANN.evaluate_at(&_Tevents, 100);
    double tru1 = _Tevents[100][0];
    double E1 = CrossEntropy(tru1, op1);
    _TANN.stash();
    
    // BP's dE/dw
    double del30 = _TANN.delta(l,n);
    std::cout << "delta_" << l << n << " = " << del30 << std::endl;
    double d30o2 = (w == _TANN.currentWeights()[l][n].size() - 1) ? del30 : del30 * _TANN.outputs()[l-1][w];
    double DE1 = d30o2;
    res.push_back(DE1);
    
    // E(w2)
    _TANN.add_to_weight(l,n,w, DW);
    double op2 = _TANN.evaluate_at(&_Tevents, 100);
    double E2 = CrossEntropy(tru1, op2);
    std::cout << "\t\t\t\t" << E2-E1 << std::endl;
    double DE2 = (E2 - E1)/DW;
    res.push_back(DE2);
    
    std::cout << "BP Delta(E) = " << DE1 << std::endl
    << "Manual Delta(E) = " << DE2 << std::endl;
    
    _TANN.revert();
    
    return res;
}*/

MVA_Analysis::MVA_Analysis(std::string cardfile, std::string events, std::string weightfile){
    // Constructor for loading pretrained NNs.
    
    // --------------------- Loading weights --------------------- //
    std::string line;
    std::ifstream infile(weightfile);
    
    if (!infile.is_open()){
        std::cout << "ERROR: cannot open file." << std::endl;
        exit(-1);
        }
    
    std::vector<int> geometry;
    
    bool geomdone = false;
    while(!geomdone){
        
        getline(infile, line);
        const char* hash = "#";
        const char* geom = "G";
        
        if (line[0] == *geom){
            std::istringstream is(line);
            std::string temp;
            is >> temp;
            while (is){
                int layersize;
                is >> layersize;
                geometry.push_back(layersize);
            }
            geomdone == true;
        }
    }
    
    weightNet weights;
    
    std::vector< Event > dummy_inputs = {std::vector<double>(1)};
    weights.push_back(dummy_inputs);
    
    for(int layer = 1; layer < geometry.size() - 1; layer++){
        std::vector< std::vector < double > > thisLayer = {};
        
        for (int node = 0; node < geometry[layer]; node++){
            std::vector< double > thisNode = {};
            getline(infile, line);
            std::istringstream is(line);
            
            for (int weight = 0; weight < geometry[layer - 1] + 1; weight++){
                double wei;
                is >> wei;
                thisNode.push_back(wei);
            }
            thisLayer.push_back(thisNode);
        }
        weights.push_back(thisLayer);
    }
    infile.close();
    // ----------------------------------------------------------- //
    
    MVA_Analysis thisAnalysis(cardfile, events, geometry);
    
    thisAnalysis._VANN.copy_weights(weights);
    thisAnalysis._TANN.copy_weights(weights);
    
    *this = thisAnalysis;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//----------------------------------------- INPUT/OUTPUT -----------------------------------------------//
//////////////////////////////////////////////////////////////////////////////////////////////////////////


void MVA_Analysis::readCard(std::string filenm){
    // Reads parameters from card
    
    pt::ptree params; 
    pt::info_parser::read_info(filenm, params);
    
    _decorrelate = params.get<bool>("decorrelate");
    _strategy = params.get<int>("strategy");
    
    _moving_avg_step_size = params.get<int>("avg_step");
    _max_ratio = params.get<double>("max_ratio");
    _sigwt = params.get<double>("sigwt");
    _bkgwt = params.get<double>("bkgwt");
    _valTrigger = params.get<double>("val_Trigg");
    
    _BPgens = params.get<int>("BP_gens");
    _GAgens = params.get<int>("GA_gens");
    
    _alpha = params.get<double>("alpha");
    
    _sweepsize = params.get<int>("BP_sweepsize");
    }


void MVA_Analysis::load_events(){
    // Reads input files into vectors
    // For each event (line) of input, function randomly
    // pushes the event onto either training or validation sets


    // -------------------- Loading from file --------------------- //
    std::string line;
    std::ifstream infile(_infile);
    
    if (!infile.is_open()){
        std::cout << "ERROR: cannot open file." << std::endl;
        exit(-1);
        }
    
    int numvars = 0;
    int lines = 0;
    int smallevs = 0;
    while(getline(infile, line)){
        
        const char* hash = "#";
        if (line[0] != *hash){
            
            // Initially, set numvars to the larger of the two
            if (lines == 1) numvars = (_rawTevents.size() > _rawVevents.size()) ? _rawTevents[0].size() : _rawVevents[0].size();
            
            bool validEv = true;
            
            std::istringstream is(line);
            int truth;
            is >> truth;
            std::string label;
            is >> label;
            double weight;
            is >> weight;
            
            std::vector<double> thisevent = {};
            double val;
            while (is >> val){
                if (val != val) { // Check for nans
                std::cout << "WARNING: nan found!" << std::endl;
                    validEv = false;
                    break;
                    }
                thisevent.push_back(val);
            }
            if (thisevent.size() < numvars){
                validEv = false;
                smallevs++;
                std::cout << "WARNING: event " lines << " (" << label << ") has too few variables: " << thisevent.size() << std::endl;
            }
            
            if (!validEv) continue;
            
            // Randomly pick if this belongs to validation or training
            std::random_device rd;
            std::default_random_engine generator(rd());
            std::uniform_real_distribution<double> uniform(0.0,1.0);
            double var = uniform(generator);
            bool validation = (var >= _valTrigger) ? true : false;
            
            if (validation){
                _rawVevents.push_back(thisevent);
                _Vlabels.push_back(label);
                _Vtruth.push_back(truth);
                _Vweight.push_back(weight);
            }
            else{
                _rawTevents.push_back(thisevent);
                _Tlabels.push_back(label);
                _Ttruth.push_back(truth);
                _Tweight.push_back(weight);
            }
            lines++;
        }
    }
    infile.close();
}

void MVA_Analysis::save_data(){
    
    std::ofstream Terrs(_path + "Terrs.txt", std::ios::app);
    std::ofstream Verrs(_path + "Verrs.txt", std::ios::app);
    std::ofstream Results(_path + "Results.dat", std::ios::app);
    std::ofstream Alphas(_path + "Alphas.dat", std::ios::app);
    
    for (int g = 0; g < _Terrs.size(); g++){
        Terrs << g << "\t" << _Terrs[g] << "\n";
        Verrs << g << "\t" << _Verrs[g] << "\n";
    }
    
    for(int line = 0; line < _Tresults.size(); line++){
        Results << _Tlabels[line];
        for (int val = 0; val < 3; val++){
            Results << " " << _Tresults[line][val];
        }
        Results << "\n";
    }
    for(int line = 0; line < _Vresults.size(); line++){
        Results << _Vlabels[line];
        for (int val = 0; val < 3; val++){
            Results << " " << _Vresults[line][val];
        }
        Results << "\n";
    }
    
    for(int a = 0; a < _alphas.size(); a++){
        Alphas << a << "\t" << _alphas[a] << "\n";
    }
    
    Terrs.close();
    Verrs.close();
    Results.close();
    
    save_weights();
}

/*void MVA_Analysis::save_decoInputs(){
    // Recording new decorrelated variables for comparison
    
    std::ofstream decoEvents(_path + "wEvents.dat", std::ios::app);
    
    for (int i = 0; i < _Tevents.size(); i++){
        decoEvents << _Ttruth[i] << "\t" << _Tlabels[i] << "\t" << _Tweight[i];
        for (int j = 0; j < _Tevents[i].size(); j++){
            decoEvents << _Tevents[i][j] << "\t";
        }
        decoEvents << "\n";
    }
    for (int i = 0; i < _Vevents.size(); i++){
        decoEvents << _Vtruth[i] << "\t" << _Vlabels[i] << "\t" << _Vweight[i];
        
        for (int j = 0; j < _Vevents[i].size(); j++){
            decoEvents << _Vevents[i][j] << "\t";
        }
        
        decoEvents << "\n";
    }
    decoEvents.close();
}*/

void MVA_Analysis::save_weights(){
    
    std::ofstream weightfile(_path + "NN_Record.dat", std::ios::app);
    
    weightNet weights = _VANN.currentWeights();
    std::vector<int> geom = _VANN.geometry();
    
    weightfile << "# NN REPORT CARD \n" << std::endl
                << "# decorrelated:\t" << _decorrelate << std::endl
                << "# strategy:\t" << _strategy << std::endl
                << "# avg_step:\t" << _moving_avg_step_size << std::endl
                << "# max_ratio:\t" << _max_ratio << std::endl
                << "# sigwt:\t" << _sigwt << std::endl
                << "# bkgwt:\t" << _bkgwt << std::endl
                << "# valtrigger:\t" << _valTrigger << std::endl
                << "# BPgens:\t" << _BPgens << std::endl
                << "# GAgens:\t" << _GAgens << std::endl
                << "# sweepsize:\t" << _sweepsize << std::endl
                << "#" << std::endl
                << "# GEOMETRY: " << std::endl
                << "#\t Inputs: " << geom[0] << std::endl;
                for (int i = 1; i < geom.size() - 1; i++){
                        weightfile << "#\t Hidden Layer " << i << ": " << geom[i] << std::endl;
                    }
    weightfile << "#\n# Best epoch: " << _VANN.best_gen() << std::endl
                << "#\n# NEURAL NETWORK WEIGHT DATA:" << std::endl
                << "\nG ";
                for (int i = 0; i < geom.size() - 1; i++){
                        weightfile << geom[i] << " ";
                    }
    weightfile << std::endl;
    
    for (int layer = 1; layer < geom.size() - 1; layer++){
        int numPrev = geom[layer-1];
        for (int node = 0; node < weights[layer].size(); node++){
            for (int wei = 0; wei < numPrev + 1; wei++){
                weightfile << weights[layer][node][wei] << " ";
            }
            weightfile << std::endl;
        }
    }
    weightfile.close();
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//---------------------------------------- PREPROCESSING -----------------------------------------------//
//////////////////////////////////////////////////////////////////////////////////////////////////////////


void MVA_Analysis::normalise_inputs(){
    // Shifts input variables to cluster around zero 
    // and resizes them to fall around the important region of 
    // the sigmoid function - i.e. +/- sqrt(3)
    
    std::cout << "Normalising inputs..." << std::endl;
    
    for (int val = 0; val < _Tevents[0].size(); val++){
        
        double total = 0;
        double min = 0;
        double max = 0;
        
        for (int ev = 0; ev < _Tevents.size(); ev++){
            double this_val = _Tevents[ev][val];
            total += this_val;
            if (this_val < min) min = this_val;
            if (this_val > max) max = this_val;
        }
            
        for (int ev = 0; ev < _Vevents.size(); ev++){
            double this_val = _Vevents[ev][val];
            total += this_val;
            if (this_val < min) min = this_val;
            if (this_val > max) max = this_val;
        }
            
        double mean = total/(_Tevents.size() + _Vevents.size());
        double range = max - min;
        
        for (int ev = 0; ev < _Tevents.size(); ev++){
            _Tevents[ev][val] -= mean;
            _Tevents[ev][val] *= (2.0*sqrt(3))/(range);
        }
        for (int ev = 0; ev < _Vevents.size(); ev++){
            _Vevents[ev][val] -= mean;
            _Vevents[ev][val] *= (2.0*sqrt(3))/(range);
        }
    }
    std::cout << "Inputs normalised." << std::endl;
}

void MVA_Analysis::build_Walker_tables(){
    // For use in the Walker Alias random sampling method
    // For info see https://en.wikipedia.org/wiki/Alias_method
    
    std::cout << "Building Walker Tables..." << std::endl;
    
    
    // Normalising event weights to taste
    int numEvents = _Tevents.size();
    double sigweightTot = 0;
    double bkgweightTot = 0;
    
    for (int ev = 0; ev < numEvents; ev++){
        if (_Ttruth[ev] == 1) sigweightTot += _Tweight[ev];
        else bkgweightTot += _Tweight[ev];
    }
    
    
    // ****** CHANGE UP TOP ******* //
    _sigScale = _sigwt / sigweightTot;         
    _bkgScale = _bkgwt / bkgweightTot;
    // **************************** //
    
    
    // *********************************** RAW TABLES *********************************** //
    
    std::vector<double> Ui = {};    // Probability
    std::vector<int> Ki = {};       // Alias
    
    std::vector< int > Ui_overs = {};
    std::vector< int  > Ui_unders = {};
    std::vector< int > Ui_exact = {};
    
    double Uevtot = 0;
    
    for (int ev = 0; ev < numEvents; ev++){
        double scale = (_Ttruth[ev] == 1) ? _sigScale : _bkgScale;
        
        double Uev = scale * _Tweight[ev] * numEvents;
        
        Uevtot += Uev;
        
        Ui.push_back(Uev);
        Ki.push_back(ev);
        
        if (Uev < 1) Ui_unders.push_back(ev);
        else if (Uev > 1) Ui_overs.push_back(ev);
        else Ui_exact.push_back(ev);
    }
    
    
    std::cout << "Raw tables built." << std::endl;
    std::cout << "Sanity check: (U_tot / n) / totalweight = " 
            << (Uevtot / numEvents)  
            << " (should be 1)"<< std::endl;
    
    
    // ********************************* REDISTRIBUTING ******************************** //
    
    std::cout << "Redistributing events..." << std::endl;
    
    while (Ui_unders.size() > 0 && Ui_overs.size() > 0){
        int unIndex = Ui_unders[0];
        int ovIndex = Ui_overs[0];
        
        // Push from over to under
        Ki[unIndex] = ovIndex;              // Alias
        Ui[ovIndex] += Ui[unIndex] - 1.0;   // Prob
        
        // Shunt under to full
        Ui_unders.erase(Ui_unders.begin());
        
        // Sort over into appropriate category
        if (Ui[ovIndex] < 1){
            Ui_unders.push_back(ovIndex);
            Ui_overs.erase(Ui_overs.begin());
        }
        else if (Ui[ovIndex] == 1){
            Ui_exact.push_back(ovIndex);
            Ui_overs.erase(Ui_overs.begin());
        }
    }
    
    while (Ui_unders.size() > 0){
        Ui[Ui_unders[0]] = 1;
        Ui_unders.erase(Ui_unders.begin());
    }
    while (Ui_overs.size() > 0){
        Ui[Ui_overs[0]] = 1;
        Ui_overs.erase(Ui_overs.begin());
    }
    
    _WalkerTable = std::make_pair(Ui,Ki);
    
    std::cout << "Walker Tables built." << std::endl;
}

void MVA_Analysis::calculate_decoMatrix(){
    // Calculates matrix which diagonalises the covariance matrix
    // For use in decorrelating variables
    
    std::cout << "Diagonalising covariance matrix..." << std::endl;
    
    int numvars = _rawTevents[0].size();
    int numTEvents = _rawTevents.size();
    int numVEvents = _rawVevents.size();
    int numEvents = numTEvents + numVEvents;
    
    //---------------------------------- Covariance Matrix ---------------------------------//
    
    // Means of each variable
    std::vector<double> xbar(numvars, 0);
    for (int ev = 0; ev < numTEvents; ev++){
        for (int var = 0; var < numvars; var++){
            xbar[var] += _rawTevents[ev][var] / numEvents;
        }
    }
    for (int ev = 0; ev < numVEvents; ev++){
        for (int var = 0; var < numvars; var++){
            xbar[var] += _rawVevents[ev][var] / numEvents;
        }
    }
    
    
    // Square matrix to fill
    MatrixXd covMat(numvars, numvars);
    
    // Filling matrix
    for (int j = 0; j < numvars; j++){
        for (int k = j; k < numvars; k++){
            double elementjk = 0;
            for (int i = 0; i < numTEvents; i++){ // sum over all events
                elementjk += (_rawTevents[i][j] - xbar[j]) * (_rawTevents[i][k] - xbar[k]);
                }
            for (int i = 0; i < numVEvents; i++){ // sum over all events
                elementjk += (_rawVevents[i][j] - xbar[j]) * (_rawVevents[i][k] - xbar[k]);
                }
            covMat(j,k) = elementjk / (numEvents - 1.0);
            covMat(k,j) = covMat(j,k);
        }
    }
    
    //------------------------------------ Eigen -----------------------------------//
    
    Eigen::EigenSolver< MatrixXd > es(covMat);
    
    // Diagonalising matrix
    _decoMatrix = es.eigenvectors().real();

    // Eigenvalue scaling matrix lambdasqrt
    VectorXd eigenvals = es.eigenvalues().real();
    
    _lambdasqrt = MatrixXd(numvars,numvars);
    for (int i = 0; i < _lambdasqrt.rows(); i++){
        for (int j = 0; j < _lambdasqrt.cols(); j++){
            if (i == j) _lambdasqrt(i,j) = 1.0 / sqrt(eigenvals(i));
            else _lambdasqrt(i,j) = 0;
            }
    }
        
    
    std::cout <<"Diagonalised." << std::endl;
}

void MVA_Analysis::decorrelate_inputs(bool validation){
    // Rotates inputs using the diagonalisation matrix
    // Should make inputs roughly decorrelated
    
    std::cout <<"Decorrelating inputs..." << std::endl;
    
    std::vector<Event>* inputEventPtr = (validation) ? &_rawVevents : &_rawTevents;
    std::vector<Event>* decoEventPtr = (validation) ? &_Vevents : &_Tevents;
    
    
    int numvars = inputEventPtr->at(0).size();
    
    // Rotating events
    for (int ev = 0; ev < inputEventPtr->size(); ev++){
        
        Event* thisevent = &inputEventPtr->at(ev);
        Event decoEvent(numvars, 0);
        
        VectorXd oldev(numvars);
        for (int i = 0; i < numvars; i++){
            oldev(i) = thisevent->at(i);
        }
        
        VectorXd newev(numvars);
        newev = _lambdasqrt * _decoMatrix.transpose() * oldev;
        
        for (int i = 0; i < numvars; i++){
            decoEvent[i] = newev(i);
        }
        decoEventPtr->push_back(decoEvent);
        
    }
    std::cout <<"Decorrelated." << std::endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//------------------------------------------------------------------------------------------------------//
//                                            TRAINING                                                  //
//------------------------------------------------------------------------------------------------------//
//////////////////////////////////////////////////////////////////////////////////////////////////////////

void MVA_Analysis::Train(int gens){
    // For details of strategies see the README
    
    if (_strategy == 0) Train_GA(gens);
    else if (_strategy == 1) Train_BackProp(gens);
    else {
        for (int reps = 0; reps < gens; reps++){
            Train_BackProp(_BPgens);
            Train_GA(_GAgens);
        }
    }
    
    std::cout << "Iteration after which error was minimised for validation sample: " << _VANN.best_gen() << "." << std::endl;
    
    // Go back to weights which minimised the error
    _VANN.revert();             
    _TANN.copy_weights(_VANN);
    
    // Record final results
    double temp = Error(_TANN, true);
    temp = Error(_VANN, true);
}


void MVA_Analysis::Train_GA(int gens){
    
    // Errors with initial weights
    _stashedTerr = Error(_TANN, false) / (1.0 - _valTrigger);
    _stashedVerr =  Error(_VANN, false) / _valTrigger;
    
    for (int gen = 0; gen < gens; gen++){
        
        _elapsedgens++;
        std::cout << "Iteration " << _elapsedgens << "(GA). TErr:" << _stashedTerr << ", VErr: " << _stashedVerr << std::endl;
        
        // Peturb weight by random amount
        _TANN.perturb(gen);
        _VANN.copy_weights(_TANN);
        
        _currentTerr = Error(_TANN, false) / (1.0 - _valTrigger);   // Error on training sample
        _currentVerr = Error(_VANN, false) / _valTrigger;           // Error on validation sample
        
        _Terrs.push_back(_currentTerr);
        _Verrs.push_back(_currentVerr);
        
        
        // DECISIONS
        if (_currentTerr <= _stashedTerr){
            // Better - keep this arrangement for next generation
            _stashedTerr = _currentTerr;
            _TANN.stash();
        }
        else _TANN.revert(); // Go back to previous arrangement
        
        if (_currentVerr <= _stashedVerr){
            // Better - save as best-so-far
            _stashedVerr = _currentVerr;
            _VANN.stash();
            _VANN.stash_gen(_elapsedgens);
        }
    }
    if (_strategy == 4){
        // Revert to best validation weights before transition to BP
        _VANN.revert();
        _TANN.copy_weights(_VANN);
    }
}

void MVA_Analysis::Train_BackProp(int gens){
    
    double Terr, Verr;
    
    for(int sweep = 0; sweep < gens; sweep++){
        _elapsedgens++;
        
        for (int evcount = 0; evcount < _sweepsize; evcount++){
            // Pick weighted random events from training sample
            
            int thisEventindex = WeightedSample();
            
            Event* thisEvent = &_Tevents[thisEventindex];
            _TANN.evaluate_at(thisEvent, _Ttruth[thisEventindex]);
            _TANN.backprop(_alpha);
            _VANN.copy_weights(_TANN);
        }
        
        // Recording error (scaled to relative sizes of samples)
        Terr = Error(_TANN,false) / (1.0 - _valTrigger);
        Verr = Error(_VANN,false) / _valTrigger;
        
        std::cout << "Iteration " << _elapsedgens << " (BP). " << "Terr: " << Terr << ", Verr: " << Verr << std::endl;
        
        _Terrs.push_back(Terr);
        _Verrs.push_back(Verr);
        _alphas.push_back(_alpha);
        
        // Decisions
        if (sweep > 1){
            // Evaluates (backwards) moving average
            int stepsize = (sweep > _moving_avg_step_size) ? _moving_avg_step_size : sweep;
            double avgVerr = 0;
            for (int i = sweep; i > sweep - stepsize; i--){
                avgVerr += _Verrs[i] / stepsize;
                }
            if (avgVerr < _stashedVerr){ // Minimum avg error so far for validation
                _VANN.stash();
                _VANN.stash_gen(_elapsedgens);
                _stashedVerr = avgVerr;
            }
        }
        
        if (Terr/_stashedTerr > _max_ratio){    // Too wildly higher
            _TANN.revert();
            _VANN.copy_weights(_TANN);
            _alpha *=0.7; // Reduce learning rate, i.e. don't step so far next time
        }
        else {
            _TANN.stash();
            // If this is better, stride further in this direction
            if (Terr < _stashedTerr) _alpha *= 1.01;
            _stashedTerr = Terr;
        }
    }
    
    if (_strategy == 4){
        // Revert to best validation weights before transition to GA
        _VANN.revert();
        _TANN.copy_weights(_VANN);
    }
    
}

double MVA_Analysis::Error(Perceptron& ANN, bool last){
    
    std::vector< Event>* infile = ANN.isValidation() ? &_Vevents : &_Tevents;
    std::vector< int>* truths = ANN.isValidation() ? &_Vtruth : &_Ttruth;
    std::vector< double>* weights = ANN.isValidation() ? &_Vweight : &_Tweight;
    
    double Error_func = 0;
    
    for (int ev = 0; ev < infile->size(); ev++){      // Iterate through each event
        
        int truth = truths->at(ev);
        double weight = weights->at(ev);
        
        // Plug into ANN
        double opinion = ANN.evaluate_at(&infile->at(ev), truth);
        double weight2 = (truth == 1) ? weight * _sigScale : weight * _bkgScale;
        if (opinion != 0. && opinion != 1.0) Error_func += CrossEntropy(truth, opinion) * weight2;
        
        if (_strategy == 0){
            // Only use if doing pure GA! Need to carry through for BP
            double besterr = ANN.isValidation() ? _stashedVerr : _stashedTerr; // (GA option)
            if (Error_func > besterr) return Error_func;    // Returns as soon as new error exceeds old
        }
        
        if (last){
            // Final evaluation - write to results
            std::vector<Event>* results = ANN.isValidation() ? &_Vresults : &_Tresults;
            Event line = {truth, weight, opinion};
            results->push_back(line);
        }
    }
    return Error_func;
}

int MVA_Analysis::WeightedSample(){
    // Uses Walker Tables to pick a weighted sample 
    
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_real_distribution<double> uniform(0.0,1.0);
    
    double x = uniform(generator);
    int n = _Tevents.size() - 1;
    
    int i = floor(n*x) + 1;
    double y = n*x + 1 - i;
    
    if (y < _WalkerTable.first[i]) return i;
    else return _WalkerTable.second[i];
}


