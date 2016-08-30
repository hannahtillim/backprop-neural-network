# Backprop Neural Network

This project uses Feed-Forward Neural Networks for classification. The analysis sorts incoming data into Training and Validation sets and uses either a genetic algorithm or a Backprop algorithm to learn the features of 'signal' (1) or 'background' (0) events, producing for each event a number between 0 and 1.  

There is functionality for input decorrelation and normalisation, adaptive learning rates and use of multiple strategies in one run.

## FILES:
MVA_main.cc  
MVA_Analysis.cc  
MVA_Analysis.h  
Perceptron.cc  
Perceptron.h  
globals.h

## REQUIRED LIBRARIES
The program uses:  
* Boost (works with v1.59.0).  
* Eigen (works with v3.2.9): Header-only library found here: http://eigen.tuxfamily.org/.

## INSTALLING
* Edit the Makefile field `EIGENDIR` to point to the directory containing the `Eigen` directory.
* Type `make`

## INPUT SYNTAX 
./analyse \<param/card/path\> \<input/data/path\> \<number of generations\> *(optional)*\<previous/weights/card/path\>

## INPUT DATA FORMAT
Data is read in from a text file. Each line is an event; the first 3 columns must contain:  
1. `truth` : `1` for a signal event, `0` for background.  
2. `source` : a label for the event source for bookkeeping purposes.  
3. `weight` : the weight given the event.  
The following columns should contain all the available kinematic variables for the event (e.g.: reconstructed Higgs mass, pT of each jet) in strict order.

## PARAM CARD
There are a variety of parameters which can affect the training. See the included cards for boilerplate and recommended values.  
* `decorrelate:` (bool) Whether to decorrelate the input variables by diagonalising the covariance matrix (recommended).
* `strategy:` (int) See below.
* `avg_step` (int) Number of generations over which the backwards running average of the validation error is taken (used to decide whether configurations are better) (BP option).
* `max_ratio:` (double) Maximum allowable ratio of current training error to previous training error before the new configuration is thrown away in favour of the previous (BP option).
* `sigwt:` (double) Weight of signal when taking weighted random samples
* `bkgwt:` (double) Same for background.  
* `val_Trigg:` (double) Percentage of events allocated to the *training* file. 
* `BP_gens:` (int) Number of BP generations to run before switching (mixed strategy option).
* `GA_gens:` (int) Same for GA (mixed strategy option).
* `alpha:` (double) Initial learning rate (BP option).
* `BP_sweepsize:` (int) Number of backprop operations carried out before measuring the error. Reduce if the NN is overtraining very quickly (BP option).

## STRATEGIES
The NN can be trained using a Genetic Algorithm, a Backprop Algorithm, or a mixture of both:  

**Option `0`: GA**: Improves by trial-and-error of random perturbations to the weights followed by an evaluation of the error.  
  * Advantages: easier to escape a local minimum.  
  * Disadvantages: Very slow, bad at digging into a nearby minimum. 

**Option `1`: BP**: Calculates optimum modification of the weights based on a gradient descent algorithm. `BP_sweepsize` decided how many operations are carried out before the error is evaluated and a decision made.  
  * Advantages: Very fast - in fact, NNs frequently overtrain within the first few generations, depending on the variations in the data. If you find this to be happening, try to reduce the `BP_sweepsize` to gain better granularity.
  * Disadvantages: Can get stuck in a local minimum even if there is a better minimum just over a rise (so to speak).

**Option `2`: Mix** (experimental) : Swaps between GA and BP in the hope that, when the BP has finished working its way into a local minimum, a few rounds of GA could help spot if there is a better configuration nearby by casting random probes. At the moment, one must specify how many generations of each are required before swapping (`BP_gens` and `GA_gens`). In this case, the 'number of generations' specified in the command line argument become the number of repetitions of one round of each (so don't set it too high).  
  * Plans to add the ability of the program to judge when the time is right to make the switch.
  * Have seen improvement in some data samples using this method, less in others. Needs optimisation.

## SAVING AND READING WEIGHTS
When the run is complete, the program will produce an `NN_Record.dat` file, which will record the parameters and results of the run. Beneath that the program will record the weights borne by the NN when the best error was obtained.  

This file can be used to load those weights at the beginning of a new run, rather than using the randomly-generated fresh weights. This can be useful for further optimisation of the weights.  

* TODO: The information will also be useful when, having decided on a final, good set of weights, one wishes to test them against another set of data without any further training.

## GRAPHING RESULTS
In the folder 'Plots' there are some python scripts to plot the data. 

1. SB_plot.py  
  * Usage: `python ./SB_plot.py Results.dat`  
  * Output: Plots of the number of signal and background events, S/sqrt(B), S/B and number of events histograms as a function of ANN decision boundary. The graph `Results_hist.png` is a good indicator - good training will result in the `0` events being pushed to the left and the `1` events to the right; the more separation the better.  
2. plot_errors.py  
  * Usage: `python ./plot_errors.py`  
  * Output: Errors for both training and validation samples as generations increase.
