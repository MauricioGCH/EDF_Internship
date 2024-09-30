"""THIS WAS CREATED AT THE BEGINNING OF THE INTERNSHIP TO ORGANIZE THE MANUALLY ANNOTATED TIMESTAMP IN A ESTABLISHED FORMAT OFR LATER TREATMENT"""

## Organizing and reading labels .xslx

import pandas as pd
import os
import glob
    ## Setting up Relative paths to be able to run on any computer
absolute_path = os.path.dirname(__file__)

relative_path = 'Dataset\\Labels'

os.makedirs(relative_path, exist_ok= True)

    
full_path = os.path.join(absolute_path, relative_path)

file_full_path = glob.glob(os.path.join(full_path,"*xlsx"))
#print(file_full_path)

headers = ['VideoName','Eel_TimeStamp','Debris_TimeStamp','Arcing_TimeStamp','DoubleTrackError_TimeStamp', 'OtherFish_TimeStamp','GT_Count','OldTP_Count',"OldFP_Count" ,"FP_Comment"]
modified_df = pd.DataFrame(columns=headers)

for xslx in file_full_path:

    df = pd.read_excel(xslx, header=None) ## opening without headers

    for i in range(len(df)): # Iteraterate row number

        row = df.iloc[i,:]
        #print(df.iloc[i,0])

        if not type(df.iloc[i,0]) is str:
            break

        ## False positives rearrangement The timestamps from the false positives will be rearranged to the correpsonding column according to the comment
        # This will not separate Timestamps rows that have different classes, i will need to manually separate them as they dont follow an easy format
        
        

        OtherFishTimeStamp = []
        DebrisTimeStamp = []
        ArcingTimeStamp = []
        TrackingTimeStamp = []
        Commentary = "Excel with no comment column"

        

        if 7 in df.columns:
            Commentary = row[7]

            #print("Commentary:", Commentary)
            if "poisson" in str(Commentary).lower() or "silure" in str(Commentary).lower():
                OtherFishTimeStamp = row[2]
                print("1 ",Commentary)
            if "arcing" in str(Commentary).lower():
                ArcingTimeStamp = row[2]
                print("2 ",Commentary)
            if "débris" in str(Commentary).lower() or "algue" in str(Commentary).lower():
            
                print("3 ",Commentary)
                DebrisTimeStamp = row[2]
            if "tracking" in str(Commentary).lower():
                TrackingTimeStamp = row[2]      
                print("4 ",Commentary)
        


        NewRow = {headers[0] : row[0], headers[1] : row[1], headers[2] : DebrisTimeStamp, headers[3] : ArcingTimeStamp, headers[4] : TrackingTimeStamp, headers[5] : OtherFishTimeStamp, headers[6] : row[4], headers[7] : row[5],headers[8] : row[6] , headers[9] : Commentary}
        
        modified_df = modified_df._append(NewRow, ignore_index=True)
        

    modified_df.fillna({'GT_Count' : 0}, inplace=True)
    modified_df.fillna({'OldTP_Count' : 0}, inplace=True)
    modified_df.fillna({'OldFP_Count' : 0}, inplace=True)
    #modified_df['GT_Count'].fillna(0, inplace=True)
    modified_df.to_excel(os.path.join(relative_path,"Organized_Labels.xlsx"), index=False)
    #print("wait")