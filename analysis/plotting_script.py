# -*- coding: utf-8 -*-
"""
Plotting script for Dynamic Routing / Templeton data

Created on Fri Jan 13 16:14:43 2023

@author: ethan.mcbride
"""

from DR_analysis_utils import Session, makePSTH, make_neuron_time_trials_tensor
from DR_analysis_utils import make_trial_da, plot_rasters, plot_heatmaps, plot_rew_nonrew_rasters
from DR_analysis_utils import plot_stim_vs_lick_aligned_rasters

# %%
def plot_data(mainPath):
    
    #hack to determine DR vs. Templeton:
    if ('Templeton' in mainPath)|('templeton' in mainPath):
        templeton_rec=True
    else:
        templeton_rec=False
        
        
    #load function
    session = Session(path=mainPath)
    session.assign_unit_areas()
    
    #make trial-based data array
    session.trial_da = make_trial_da(session.good_units, session.spike_times, session.trials)
    
    # #plot heatmaps function
    # plot_heatmaps(mainPath, session.trial_da, session.trials, session.good_units, templeton_rec)
    
    # # #plot rasters function
    # plot_rasters(mainPath, session.trial_da, session.good_units, session.trials, templeton_rec)
    
    # # #plot rewarded vs. unrewarded rasters
    # plot_rew_nonrew_rasters(mainPath, session.trial_da, session.good_units, session.spike_times, 
    #                         session.trials, session.lick_times, templeton_rec)
    
    
    plot_stim_vs_lick_aligned_rasters(session, mainPath, templeton_rec)
    
    
    
    
    session=None


# %% run loop on experiment folders

mainPaths = [
    #sound pilot
    # r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\625820_06222022\2022-06-22_14-25-10\processed",
    # r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\625821_07112022\2022-07-11_14-42-15\processed",
    # r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\625821_07122022\2022-07-12_13-51-39\processed",
    
    #opto pilot
    # r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\opto pilot\2022-11-07_12-31-20_635891\processed",
    # r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\opto pilot\2022-11-08_11-03-58_635891\processed",
    # r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\opto pilot\2022-11-14_13-18-05_636760\processed", #error w/ vsync/photodiode?
    # r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\opto pilot\2022-11-15_14-02-31_636760\processed", #error w/ vsync/photodiode?
    
    #templeton pilot
    # r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\2022-07-26_14-09-36_620263\processed",
    # r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\2022-07-27_13-57-17_620263\processed",
    # r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\2022-08-02_15-40-19_620264\processed",
    # r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\2022-09-19_13-48-26_628801\processed",
    # r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\2022-09-26_12-48-09_636397\processed",
    # r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\2022-09-27_11-37-08_636397\processed",
    # r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\2022-12-05_13-08-02_644547\processed",
    # r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\2022-12-06_12-35-35_644547\processed",
    # r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\2023-01-17_11-39-17_646318\processed",
    # r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\2023-01-18_10-44-55_646318\processed",
    # r"Y:\2023-02-27_08-14-30_649944\processed",
    # r"Y:\2023-02-28_09-33-43_649944\processed",
    
    #DR pilot
    r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\Task 2 pilot\DRpilot_626791_20220815\processed",
    r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\Task 2 pilot\DRpilot_626791_20220816\processed",
    r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\Task 2 pilot\DRpilot_626791_20220817\processed",
    
    # r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\Task 2 pilot\DRpilot_636766_20230123\processed",
    # r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\Task 2 pilot\DRpilot_636766_20230124\processed",
    # r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\Task 2 pilot\DRpilot_636766_20230125\processed",
    
    # r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\Task 2 pilot\DRpilot_644864_20230201\processed",
    
    # r"Y:\DRpilot_644867_20230220\processed",
    # r"Y:\DRpilot_644867_20230221\processed",
    # r"Y:\DRpilot_644867_20230222\processed",
    # r"Y:\DRpilot_644867_20230223\processed",
    ]

for mm in mainPaths[:]:
    plot_data(mm)