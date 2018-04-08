import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
import os
from glob import glob

training_csv_file = 'boneage-training-dataset.csv'
training_dir = 'boneage-training-dataset'
dataset = pd.read_csv(training_csv_file)
dataset['gender'] = dataset['male'].map(lambda x: "male" if x else "female")
dataset['path'] = dataset['id'].map(lambda x: training_dir+"/"+str(x)+".png")
dataset['exists'] = dataset['path'].map(os.path.exists)

print(str(sum([1 if i else 0 for i in dataset['exists']]))+" images found out of "+str(len(dataset['exists']))+" images.")
dataset = dataset[dataset.exists]

dataset[['boneage','male']].hist(figsize = (10, 5))

age_groups = 4

dataset['age_range'] = pd.qcut(dataset['boneage'],age_groups)
sample_dataset = dataset.groupby(['boneage','male']).apply(lambda x: x.sample(1)).reset_index(drop=True)

fig,m_axes = plt.subplots(age_groups,2)
for c_ax, (_,c_row) in zip(m_axes.flatten(),sample_dataset.sort_values(['age_range','gender']).iterrows()):
	c_ax.imshow(imread(c_row['path']))
	c_ax.axis('off')
	c_ax.set_title('{boneage} months, {gender}'.format(**c_row))
plt.show()