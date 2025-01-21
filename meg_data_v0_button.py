# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:11:07 2024

@author: lpxaj7

code to check the button responses
1=100ms (blue/left)
2=200ms
3=300ms
4=400ms (red/ right)

note to self: u need to align the data with et and find rec block last and then subtract time values u get here to get actual time to match with responses
"""
import re
import os
import mne
from mne.datasets.eyelink import data_path
from mne.preprocessing.eyetracking import read_eyelink_calibration
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path1='C:/Users/lpxaj7/OneDrive - The University of Nottingham/Documents/Experiments/Year_1/v0/DrivingSim/'

#MEG
filename_meg= '15463_DA_meg.fif'
read_file_meg=os.path.join(path1,'fifConverter/',filename_meg)

filename_events='15463_DA_events_eve.fif'
read_file_events=os.path.join(path1,'fifConverter/',filename_events)

savefigpath='C:/Users/lpxaj7/OneDrive - The University of Nottingham/Documents/Experiments/Year_1/v0/analysis/sim/response/'  
#%%
filename_split= re.split(r'[_\.]',filename_meg)
participant= int(filename_split[0])
#%%
# Loading data

#MEG
raw_meg=mne.io.read_raw_fif(read_file_meg,preload=True)
print(raw_meg.info)
original_raw_meg=raw_meg.copy()
scalings = {'mag':1e-11,'stim':3}  
raw_meg.plot(scalings=scalings)
raw_meg.copy().pick_types(meg=False, stim=True).plot()

#%%
df_meg = pd.DataFrame(data=raw_meg[['A11','A2','A9','A15','A6','A13','A4']][0].T,\
                      index=raw_meg.times,\
                          columns=['x_pos_initial','y_pos_initial','pupil_size',\
                                   'gas','brake','steering','button press'])

#%%
#plot data for both meg and et x pos
def plotfigure(plt1,plt2,xlabel,ylabel,legend,title):
    #variable to plot
    fig=plt.figure(figsize=(15, 6))  # Create a new figure
    
    # Set common properties for both subplots
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset']='cm'
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams.update({'font.size': 17})
    
    plt.plot(plt1)
    plt.plot(plt2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(legend)
    plt.tight_layout()
    plt.title(title)
    plt.show()
    return fig

#%% plot to see gas channel
button_me=raw_meg['A4']
plt.plot(button_me[1], button_me[0].T)

plt.show()
#%%

button= pd.DataFrame(button_me[0].T,columns=['press_voltage'])
#find difference of data
button_diff= button.diff()
#create a column 'label', and give value =1 if the difference is greater tha |2|, to find peaks in the data
button_diff['label']=0
for i in range(len(button_diff)):
    if abs(button_diff.iloc[i,0]) >2:
        button_diff.iloc[i,1]=1
button['label']=button_diff['label']
button.index=button_me[1]

#%%
# Find the indices where the value is 1
indices= button.index[button['label']==1]

button['response']=0

# Iterate over the indices and find time differences to find the reponse buttons
for i in range(len(indices)-1):
    diff=( indices[i+1]-indices[i])*1000
    print(diff)
    if 100-5 <= diff <= 100+5:
        # Label both locations as 1
       button.loc[indices[i], 'response'] = 1
       #button.loc[indices[i+1], 'names'] = 1
    elif 200-5 <= diff <= 200+5:
        # Label both locations as 2
        button.loc[indices[i], 'response'] = 2
        #button.loc[indices[i+1], 'names'] = 2
    elif 300-5 <= diff <= 300+5:
        # Label both locations as 2
        button.loc[indices[i], 'response'] = 3
        #button.loc[indices[i+1], 'names'] = 3
    elif 400-5 <= diff <= 400+5:
        # Label both locations as 2
        button.loc[indices[i], 'response'] = 4
        #button.loc[indices[i+1], 'names'] = 4
#%%
button=button.drop(columns=['label'])    

#%% save button responses as csv
Response_file=savefigpath+ str(participant)+'_Response'
button.to_csv(f"{Response_file}.csv", index=True)
#%%
plt.plot(button_diff)
plt.show()


#%%events
raw_events= mne.read_events(read_file_events)
events_button=mne.find_events(raw_meg, stim_channel='A4')

fig= mne.viz.plot_events(events_button,sfreq=raw_meg.info['sfreq'],first_samp=raw_meg.first_samp)
#%%
gas=raw_meg['A13']
plt.plot(gas[1], gas[0].T)

plt.show()