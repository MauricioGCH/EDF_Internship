

import os
import glob
import pandas as pd
import shutil

base_path = r'NewMasks2'


ListOfvideos = glob.glob(r'NewMasks2\2014*')

for i in range(len(ListOfvideos)):
    
    original = glob.glob(os.path.join(ListOfvideos[i],"Original","t*"))
    foreground = glob.glob(os.path.join(ListOfvideos[i],"Foreground","t*"))

    for j in range(len(original)):
        print("--------")
        print(len(glob.glob(os.path.join(original[j], "*"))))
        print(len(glob.glob(os.path.join(foreground[j], "*"))))
        print("--------")
        assert len(glob.glob(os.path.join(original[j], "*"))) == len(glob.glob(os.path.join(foreground[j], "*")))
   