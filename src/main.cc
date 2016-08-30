#include "globals.h"
#include "MVA_Analysis.h"
#include <random>


int main(int argc, char **argv)
{
    if (argc < 4) {
        std::cout << "Not enough arguments. Correct format: ./main.cc <param card path> <data file path> <number of generations> (optional) <weight card path>" << std::endl;
        return 0;
        }
        
    std::cout << std::endl
        << "////////////////////////////////////////////////" << std::endl
        << "//--------------------------------------------//" << std::endl
        << "//           BACKPROP NEURAL NET              //" << std::endl
        << "//--------------------------------------------//" << std::endl
        << "////////////////////////////////////////////////" << std::endl
        << std::endl;
    
    
    //--------I/O--------//
    std::string cardnm = argv[1];
    std::string infilenm = argv[2];
    std::string weightcardnm;
    bool weightcard = false;
    if (argc == 5){
        weightcard == true;
        weightcardnm = argv[4];
    }
    int gens = atoi(argv[3]);
    std::vector<int> geom = {5,3,1};
    
    std::cout << "INPUT FILE: \t\t" << infilenm << std::endl
    << "BACKPROP GENERATIONS: \t" << gens << std::endl;
    
    if (weightcard){
        MVA_Analysis analysis(cardnm, infilenm, weightcardnm);
        
        //---------------------------------Training---------------------------------//
    
        std::cout << "//--------------------------------------------//" << std::endl
                    << "BEGINNING TRAINING..." << std::endl;
        analysis.Train(gens);
    
        //--------------------------------------------------------------------------//
    
        analysis.save_data();
    
        std::cout << "//--------------------FIN---------------------//" << std::endl;
        
    }
    else {
        MVA_Analysis analysis(cardnm, infilenm, geom);
    
        //---------------------------------Training---------------------------------//
    
        std::cout << "//--------------------------------------------//" << std::endl
                    << "BEGINNING TRAINING..." << std::endl;
        analysis.Train(gens);
    
        //--------------------------------------------------------------------------//
    
        analysis.save_data();
    
        std::cout << "//--------------------FIN---------------------//" << std::endl;
    }

    
}
