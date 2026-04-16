import sys 


from pathlib import Path
# .parent gets the directory containing the file
script_dir = Path(__file__).resolve().parent
sys.path.insert(0,script_dir)
import Pooling_Layers
import numpy as np 

def Pool_Data( Data , poolsize, stride) :
    X =[]
    for i in range(len(Data)) :
        x = Pooling_Layers.max_pooling_2d( Data[i] , poolsize , stride )
        X.append(x)
    return np.array(X)
