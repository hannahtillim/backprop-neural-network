#include "globals.h"
#include "MVA_Analysis.h"
#include <random>

int main(int argc, char **argv)
{
    std::cout << "BACKPROP NEURAL NET" << std::endl;
    
    if (argc < 3) {
        std::cout << "Not enough arguments. Correct format: ./main.cc <data file path> <number of generations>" << std::endl;
        return 0;
        }
    
    std::string infilenm = argv[1];
    int BPgens = atoi(argv[2]);
    std::vector<int> geom = {5,3,1};
    
    std::cout << "Input data: \t\t" << infilenm << std::endl
    << "BACKPROP GENERATIONS: \t" << BPgens << std::endl;
    
	MVA_Analysis resolved(infilenm, geom);
    
    std::vector<double> multiples = {};
    
    std::vector<double> geom2 = {15,5,3,1};
    
/*    //-----------------------------Backprop test -----------------------------//
    std::ofstream file;
    file.open(resolved.path() + "BP_test_BP.txt");
    
    std::ofstream file2;
    file2.open(resolved.path() + "BP_test_Manual.txt");
    
    
    double Dw = 1;
    while (Dw > 0.00001){
        std::cout << "\t\t\t\t Dw:" << Dw << std::endl;
        std::vector<double> res = resolved.BP_test(1, 3, 7, Dw);
        file << Dw << "\t" << res[0] << "\n";
        file2 << Dw << "\t" << res[1] << "\n";
        Dw /= 1.1;
    }
    file.close();
    file2.close();
    //--------------------------------------------------------------------------//
    */

    //---------------------------------Training---------------------------------//
    
    std::cout << "Beginning training..." << std::endl;
    resolved.Train_BackProp(BPgens);
    
    //--------------------------------------------------------------------------//
    
    
    
    resolved.save_data();
}
