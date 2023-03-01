# -*- coding: utf-8 -*-
"""
DR Processing Script:
    
    Script for processing Dynamic Routing and Templeton ephys recordings
    
    Add full paths to base experiment session directories in 'mainPaths' list

Created on Fri Jan 13 15:22:03 2023

@author: ethan.mcbride
"""

import numpy as np
import os
import glob

from DR_processing_utils import load_behavior_data, load_rf_mapping, sync_data_streams 
from DR_processing_utils import align_trial_times, align_rf_trial_times
from DR_processing_utils import align_spike_times, load_lick_times

#only for sound pilot recordings
from DR_processing_utils import load_sound_pilot_data, sync_data_streams_sound_pilot

# %%
def process_ephys_sessions(mainPath, mouseID, exp_num):
    
    #hack to account for RF mapping order:
    if ('636397' in mainPath) | ('635891' in mainPath) | ('636760' in mainPath) | ('636766' in mainPath) | ('644864' in mainPath):
        RF_first=True
    else:
        RF_first=False
        
    if ('625820' in mainPath) | ('625821' in mainPath):
        sound_pilot=True    
    else:
        sound_pilot=False
        
    behavPath = glob.glob(os.path.join(mainPath, 'DynamicRouting*.hdf5'))[0] #assumes that behavior file is the only .hdf5!
    rfPath = glob.glob(os.path.join(mainPath, 'RFMapping*.hdf5'))
    ephysPath = glob.glob(os.path.join(mainPath,'Record Node*','experiment*','recording*'))
    if len(ephysPath)==0:
        ephysPath = glob.glob(os.path.join(mm,'*_'+mouseID+'*','Record Node*','experiment*','recording*'))[0]
    elif len(ephysPath)==1:
        ephysPath = ephysPath[0]
    
    h5syncPath = glob.glob(os.path.join(mainPath, '*.h5'))
    syncsyncPath = glob.glob(os.path.join(mainPath, '*.sync'))
    
    if len(h5syncPath)>0:
        syncPath = h5syncPath[0] #assumes that sync file is the only .h5!
    elif len(syncsyncPath)>0:
        syncPath = syncsyncPath[0]
        
    processedDataPath = os.path.join(mainPath,'processed')
    
    if os.path.isdir(processedDataPath)==False:
        os.mkdir(processedDataPath)
    
    if sound_pilot:
        trials_df, trialSoundArray, trialSoundDur, soundSampleRate, deltaWheelPos, startTime \
            = load_sound_pilot_data(behavPath)
            
        syncData, probeNames, probeDirNames = sync_data_streams_sound_pilot(syncPath,ephysPath)
        
    else:
        trials_df, trialSoundArray, trialSoundDur, soundSampleRate, deltaWheelPos, startTime \
            = load_behavior_data(behavPath)
    
        syncData, probeNames, probeDirNames = sync_data_streams(syncPath,ephysPath)
    
    trials_df, frames_df = align_trial_times(trials_df, syncData, syncPath, ephysPath, trialSoundArray, 
                                  trialSoundDur, probeNames, probeDirNames, soundSampleRate, 
                                  deltaWheelPos, RF_first)
    
    unitData_df, spike_times, mean_waveforms = align_spike_times(ephysPath, syncData, probeNames, probeDirNames, startTime, mouseID, exp_num)
    
    lick_times = load_lick_times(syncPath)
            
    #RF mapping
    if len(rfPath)>0:
        rfPath = rfPath[0]
        rf_df, rf_trialSoundArray, rf_soundDur, rf_deltaWheelPos = load_rf_mapping(rfPath)
        rf_df, rf_frames_df = align_rf_trial_times(rf_df, syncData, syncPath, ephysPath, rf_trialSoundArray, 
                                                   rf_soundDur, soundSampleRate, rf_deltaWheelPos, RF_first)
        
        rf_df.to_csv(os.path.join(processedDataPath,'rf_mapping_trials.csv'))
        rf_frames_df.to_csv(os.path.join(processedDataPath,'rf_mapping_frames.csv'))
    
    
    ##Save individual files for each type of data
    np.save(os.path.join(processedDataPath,'spike_times_aligned.npy'),spike_times,allow_pickle=True)
    np.save(os.path.join(processedDataPath,'mean_waveforms.npy'),mean_waveforms,allow_pickle=True)
    np.save(os.path.join(processedDataPath,'lick_times.npy'),lick_times,allow_pickle=True)
    
    unitData_df.to_csv(os.path.join(processedDataPath,'unit_table.csv'))
    trials_df.to_csv(os.path.join(processedDataPath,'trials_table.csv'))
    frames_df.to_csv(os.path.join(processedDataPath,'frames_table.csv'))

# %% run loop on experiment folders

mainPaths = [
    #sound pilot
    # r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\625820_06222022\2022-06-22_14-25-10",
    # r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\625821_07112022\2022-07-11_14-42-15",
    # r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\625821_07122022\2022-07-12_13-51-39",
    
    #opto pilot
    # r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\opto pilot\2022-11-07_12-31-20_635891",
    # r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\opto pilot\2022-11-08_11-03-58_635891",
    # r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\opto pilot\2022-11-14_13-18-05_636760", #error w/ vsync/photodiode?
    # r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\opto pilot\2022-11-15_14-02-31_636760", #error w/ vsync/photodiode?
    
    #templeton pilot
    # r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\2022-07-26_14-09-36_620263",
    # r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\2022-07-27_13-57-17_620263",
    # r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\2022-08-02_15-40-19_620264",
    # r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\2022-09-19_13-48-26_628801",
    # r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\2022-09-26_12-48-09_636397",
    # r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\2022-09-27_11-37-08_636397",
    # r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\2022-12-05_13-08-02_644547",
    # r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\2022-12-06_12-35-35_644547",
    # r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\2023-01-17_11-39-17_646318",
    # r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\2023-01-18_10-44-55_646318",
    
    #DR pilot
    r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\Task 2 pilot\DRpilot_626791_20220815",
    r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\Task 2 pilot\DRpilot_626791_20220816",
    r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\Task 2 pilot\DRpilot_626791_20220817",
    # r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\Task 2 pilot\DRpilot_636766_20230123",
    # r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\Task 2 pilot\DRpilot_636766_20230124",
    # r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\Task 2 pilot\DRpilot_636766_20230125",
    # r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\Task 2 pilot\DRpilot_644864_20230201",
    ]

exp_nums = [
    #sound pilot
    # 1, #625820
    # 1,2, #625821
    
    #opto pilot
    # 1,2, #635891
    # 1,2, #636760
    
    #templeton pilot
    # 1,2, #620263
    # 1, #620264
    # 1, #628801
    # 1,2, #636397
    # 1,2, #644547
    # 1,2, #646318
    
    #DR pilot
    1,2,3, #626791
    # 1,2,3, #636766
    # 3 #644864
    ]

for im,mm in enumerate(mainPaths[:]):
    mouseID=[x for x in mm.split('_') if len(x)==6 and x.isdigit()][0]
    process_ephys_sessions(mm, mouseID, exp_nums[im])
    
