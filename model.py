#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install fastcore')
get_ipython().system('pip install fastai')


# In[2]:


from fastcore.all import *
import time
from fastai.vision.all import *
import pandas as pd
import numpy as np
import os


# In[3]:


searches = 'Glaucoma', 'Cataracts', 'Uveitis', 'Crossed_Eyes', 'Bulging eyes'

# change the path
path = '/Users/taimourabdulkarim/Desktop/Fiverr/Reachsumim/Augmented Dataset'
data_dir_list = os.listdir(path)
print(data_dir_list)


# In[ ]:


# remove the .DS_Store from the list
data_dir_list.remove('.DS_Store')
print(data_dir_list)


# In[ ]:


get_items = get_image_files(path)


# In[ ]:


dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path)

dls.show_batch(max_n=10)


# In[ ]:


dls.vocab


# In[ ]:


blocks = (ImageBlock, CategoryBlock),
get_items = get_image_files,
splitter = RandomSplitter(valid_pct=0.2, seed=42),
dls.show_batch()


# In[ ]:


learn = vision_learner(dls, resnet34, metrics=accuracy)
learn.fine_tune(50)


# In[ ]:


learn.summary()


# In[ ]:


learn.show_results()


# In[ ]:


learn.lr_find()


# In[ ]:


import matplotlib.pyplot as plt

# Set font size for the entire plot
plt.rcParams['font.size'] = 7

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(20)
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()


# In[ ]:




