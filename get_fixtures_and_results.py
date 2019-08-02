import numpy as np
import os
import csv
import pandas as pd
import sys
from get_api import *

' Run '
if __name__ == "__main__":

    if len(sys.argv[1].split("-")) == 1:
	    gw = int(sys.argv[1])
    else:
        fgw = (sys.argv[1]).split("-")[0]
        sgw = (sys.argv[1]).split("-")[1]
        gw = np.linspace(int(fgw), int(sgw), int(sgw) - int(fgw) + 1).astype(int)
    
    getFixtures(gw)