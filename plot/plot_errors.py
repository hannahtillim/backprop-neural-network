from matplotlib import pyplot;
from pylab import genfromtxt; 
  
mat0 = genfromtxt("Terrs.txt");
mat1 = genfromtxt("Verrs.txt");
 
pyplot.plot(mat0[:,0], mat0[:,1], label = "Training");
pyplot.plot(mat1[:,0], mat1[:,1], label = "Validation");
 
pyplot.legend();
pyplot.savefig("Errors.png")
pyplot.show();
