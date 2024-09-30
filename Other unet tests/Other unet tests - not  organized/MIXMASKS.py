##TODO function to put as one mask when there are several obj in an image
import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
start = 631
finish = 672
numbers = list(range(start, finish+1))
basename = []
for i in range(len(numbers)):
    paths = glob.glob(os.path.join(r'Small_ARIS_Mauzac_UnetReadyCopy\Test\2014-11-05_184000\Foreground\t5',str( "*"+ str(numbers[i]))+"*"))
    #basename.append(os.path.basename(paths[i][0]))

    

    FinalMask = []
            #mix
    if not paths:
        continue
    for j in paths: 
        image = cv2.imread(j)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _,image = cv2.threshold(image,127,255, cv2.THRESH_BINARY)
        FinalMask.append(image)

    Mask = np.zeros_like(FinalMask[0])
    for j in range(len(FinalMask)):
        Mask = Mask + FinalMask[j]
    Mask = np.where(Mask != 0, 255, Mask)
    cv2.imwrite(os.path.join(r'Small_ARIS_Mauzac_UnetReadyCopy\Test\2014-11-05_184000\Foreground\t5\tmix',str("m_SmallFish_2014-11-05_184000_t3_Obj_frame"+ str(numbers[i])+ ".jpg")), Mask)
    plt.imshow(Mask, cmap ="gray")
    plt.title("Mix Mask")
    plt.axis("off")  # Turn off axis labels
    plt.show()
    plt.close()  # Clear the plot
    
                




