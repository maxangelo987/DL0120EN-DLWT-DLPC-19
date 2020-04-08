###############################################################
#LINEAR-REGRESSION
###############################################################

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
'exec(%matplotlib inline)'
plt.rcParams['figure.figsize'] = (10, 6)


X = np.arange(0.0, 5.0, 0.1)
print(X)

##You can adjust the slope and intercept to verify the changes in the graph
a = 1
b = 0

Y= a * X + b 

plt.plot(X, Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()


