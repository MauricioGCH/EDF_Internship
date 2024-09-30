import pandas as pd 
import json
import os
import glob
import numpy as np

df = pd.read_csv(r'C:\Users\d42684\Documents\STAGE\CODES\segment-anything-main\notebooks\SecondVersion_Annotations_Filled.csv')

filtered_df0 = df[df['file_attributes'] != '{"Object_Count":"0"}']

ImageList = glob.glob(os.path.join('C:\\Users\\d42684\\Documents\\STAGE\\CODES\\ACtoolbox-main\\Dataset\\Small_ARIS_Mauzac\\TEST\\All_Originals\\*.jpg'))
for imagepath in ImageList:

    imageName = imagepath.split('\\')[-1]


    filtered_df = filtered_df0[filtered_df0['filename'] == imageName]

    if filtered_df.empty:
        continue  # Skip this iteration if no matches found
    

    #info = json.loads(filtered_df['region_shape_attributes'])

    cx = []
    cy = []

    for i in range(len(filtered_df)):

        info = json.loads(filtered_df.iloc[i]['region_shape_attributes'])

        if info['name'] == 'point':

            cx.append(info['cx'])
            cy.append(info['cy'])
            print(f"Point - cx: {cx}, cy: {cy}")

        elif info['name'] == 'rect':
            # Process rectangle information
            x0 = info['x']
            y0 = info['y']
            width = info['width']
            height = info['height']
            print(f"Rectangle - x: {x0}, y: {y0}, width: {width}, height: {height}")
            print('wait')
    array = np.column_stack((cx, cy))