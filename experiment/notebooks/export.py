# %%
# Data exploration with the deep learning dataset
import pandas as pd

df = pd.read_csv('/home/madhav/workstation/Lab/experiment/data/DL-PROJECT-DATASET/train.csv')

df.head()

# %%
# Dataset foundational inspection
""" 
Structure of dataframe
image_id,defect_type, mask_rle, dimension
img_1, defect_a,1193 3203
img_1, defect_b,,dim
img_1, defect_c,,dim
img_1, defect_d,,dim

If a image have a defect rle would be not None,
or if it have it would have the value

""" 

# NaN values fill for extraction
df['mask_rle'] = df['mask_rle'].fillna(0)

zero_count = df[
    df['mask_rle'] == 0
].groupby('image_id').size() # series aggrigation

count = (zero_count == 4).sum()
total = (len(df) / 4)

print(f'Total Images : {total} \n Clean Images : {count} \n ratio : {(count/total)*100}')

# %% [markdown]
# # Foundational data inspection report
# 
# With the given dataset, with simple distribution plot about defects, we can observe that most abundant defect is type c, which dominates all other categories into total image count, we can go further with exploring overlaping features set through this dataset.
# 
# Inspections
# 1. Count for each defects : defect c dominates
# 2. With solo defect occurance count

# %%
# Defects distribution inspection
import matplotlib.pyplot as plt 
import pandas as pd

fig, axs = plt.subplots(1,4, figsize=(18,4))

# How many images consist of defect_a,b,c,d ...
""" 
for each of the defect simply filter length with condition
of their rle being not zero
"""
defects = list(df['defect_type'].unique())
defect_info = {}
for defect in defects:
    defect_count = df[
        df['defect_type'] == defect
    ]['mask_rle'] != 0
    # len used for counting total | sum would get the positives added
    defect_count = defect_count.sum()
    defect_info[defect] = defect_count

x = defect_info.keys()
y = defect_info.values()

axs[3].bar(
    x=x,
    height=y,
    color='forestgreen',
    edgecolor='black'
)

axs[3].set_title('DEFECT DSITRIBUTION')

# INVESTIGATING : DEFECTS DISTRIBUTION
""" 
Switch view through
== 0 > no defect class
!= 0 > Defect containing
"""
with_defects = df[
    df['mask_rle'] != 0
]

# Listing total occurance of each images in filter with_defects
values = with_defects['image_id'].value_counts()

def defect_distribution(values):
    ''' 
    parameters
    values
        A frequency dictionary about images fetched from root df
        containing either images with defect, or clean images

        with defect filter could have images with more then one occurance
        filtered for ploting in functions internal logic
    '''
    for i in range(1, 4):
        exclusive_defects = values[
            values == i
        ] #  Tweak for differnt view about class distribution for the dataset

        exclusive_defect_image_count = len(exclusive_defects)
        print(f"Images with exclusive {i} defect occurance count : {exclusive_defect_image_count}")

        defect_images = with_defects[
            with_defects['image_id'].isin(list(
                exclusive_defects.index
            ))
        ]

        defect_images['defect_type'].value_counts().plot(kind='bar', title=f'Exclusive distribution with {i} defects', ax=axs[i-1], edgecolor='black', color='royalblue')
defect_distribution(values)

# %%
# images with more then one defect count
exclusive_defects = values[
    values != 3
]

exclusive_defect_image_count = len(exclusive_defects)
print(f"Images with more then one defect : {exclusive_defect_image_count}")

# Images with no defect
total_images = list(df['image_id'].unique())
count = 0
for image in total_images:
    if not(image in values):
        count += 1
count, len(total_images)

# %%
# Image metadata indexing

from pathlib import Path

root_dir = Path(
    '/home/madhav/workstation/Lab/experiment/data/DL-PROJECT-DATASET/train'
)

meta_information = []

for path in root_dir.iterdir():

    if not path.is_file():
        continue

    meta_information.append({
        'image_id': path.stem,
        'image_path': str(path)
    })

# Image ID-PATH indexing data frame
meta_information = pd.DataFrame(meta_information)

print(f'Total image files: {len(meta_information)}')

# %% [markdown]
# ## Spatial exclusivity and output head
# 
# With the final network emiting its ouput channel, we need to be sure which final output head to use, we use softmax where only one possibility could take place out of all, and use logistic based if multiple could happen at once,
# 
# ```md
# With insepcting the source dataset with pixel level exclusivity we can conclude with this deccission, if one pixel is occupied with more then one defect class then  we would move fro logistics based output head or softmax ( probabilly not)
# ```

# %%
# Inspecting images from the defects perspective

import random
import torchvision
from PIL import Image

defect_number = 1 # listing images having more then one defect
target_images = list(values[
    values >= defect_number
].index)

# Taking sample for inspecting images
sample_images = random.sample(target_images, 10)

fig, axs = plt.subplots(10,1, figsize=(10,30))
i = 0
# Fetching path information for target images for plot
for img in sample_images:
    path = list(meta_information[(meta_information['image_id'] == img)]['image_path'])[0]
    
    name = img
    img = Image.open(path)
    axs[i].imshow(img)
    axs[i].set_title(f'ID: {name}')
    i += 1
    

# %%
encoding = list(df[
    (df['mask_rle'] != 0) & (df['defect_type'] == 'defect_d')
]['mask_rle'])

encoding

# %%
# Decoder pathway for the rle

encoding = list(df[
    (df['mask_rle'] != 0) & (df['defect_type'] == 'defect_a')
]['mask_rle'])

pick = random.randint(0, len(encoding)-1)
encoding = encoding[pick]

# RUN LENGTH ENCODING | Column major
import numpy as np

""" 
Making decoder for rle

Position is string which itself needs to be encoded for position
thus extract the position through the index via taking 128 as the upper limmit to extract
the final position for maping
""" 


splits = encoding.split(' ')
direct_map = np.zeros((128, 800))

# Position extraction logic
def extract_position(position_string):
    ''' 
    input : position_string

    Logic
    + with len 4 -> check if position accuired
    + odd position
        take first 2 -> x
        rest for y if less then 800
            check one more for x
            if less then 128 take that split
                reject it
    with x capped at 128 , y capped with 800

    returns : 
    None if its not valid -> for filteration
    tuple with integer number for position

    1819 : working full pipe :)
    '''
    x,y = 10000, 10000 # Initialization

    if len(position_string) % 2 == 0:
        split_index = int(len(position_string) / 2)
        x = int(position_string[:split_index])
        y = int(position_string[split_index:])
    else:
        x = int(position_string[:2])
        y = int(position_string[2:])

        # Internal odd check layer
        if y > 800:
            x = int(position_string[:3])
            y = int(position_string[3:])
        
    # Limmit check
    if (x <= 128) and (y <= 800):
        return (x,y)
    else:
        print(f'Position check crashed with : {position_string}')
        return None
        


# Building position and run pair
info = []

encode_length = len(splits)
for pos_i in range(0, encode_length, 2):
    
    position_string = splits[pos_i]
    run = int(splits[pos_i + 1])

    # Extract position from position string
    position = extract_position(position_string)
    if position == None:
        print(f"Might be the case this rle is corrupted ")
        break
    element = (position, run)
    info.append(element)

print(info) # Final maping to plot with

# HACK : With the indexing problem being solved still outflow is happening winth run and position at indexing
made = True
final_map = np.zeros((128, 800))
for elem in info:
    position, run = elem
    x,y = position
    starting_index = (x,y)
    end_index = (starting_index[0] + run, starting_index[1] + run)
    print(starting_index, end_index)
    try:
        final_map[starting_index, end_index] += 1
    except IndexError as e:
        print(f"Over flow event detected with index information : {starting_index, end_index}")
        made = False

if made:
    print(f"final maping made : 🎉 : {pick}")

# %%
plt.figure(figsize=(40,3))
plt.imshow(final_map)


