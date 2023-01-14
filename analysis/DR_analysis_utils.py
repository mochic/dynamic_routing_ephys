# -*- coding: utf-8 -*-
"""
DR Analysis Utils

Useful functions for Dynamic Routing / Templeton analysis

Created on Mon Jan  9 11:16:14 2023

@author: ethan.mcbride
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import patches
import os
import glob


#Simple session class for analyzing DR data
class Session:
    '''
    Simple Session object for convenient analysis of ephys data
    INPUT:
        path: full path to data processed by DR_processing_script
    OUTPUT:
        session object containing all relevant session data
        session.trials: pandas DataFrame with information about each trial
        session.units: DataFrame with info about each unit
        session.good_units: DataFrame with info about units, filtered by quality metrics
        session.spike_times: dictionary of spike times, with each key corresponding to a unit_id in the units table
        session.lick_times: list of all recorded lick times during the session
        session.frames: table with time of each frame, along with running speed
    '''
    def __init__(self,params=None,path=None):
        
        #load trials
        self.trials=pd.read_csv(os.path.join(path,'trials_table.csv'))
        trial_stim_dur=np.zeros(len(self.trials))
        for tt in range(0,len(self.trials)):
            if self.trials['trial_sound_dur'].iloc[tt]>0:
                trial_stim_dur[tt]=self.trials['trial_sound_dur'].iloc[tt]
            elif self.trials['trial_vis_stim_dur'].iloc[tt]>0:
                trial_stim_dur[tt]=self.trials['trial_vis_stim_dur'].iloc[tt]

        self.trials['trial_stim_dur']=trial_stim_dur
        
        #load lick times
        self.lick_times=np.load(os.path.join(path,'lick_times.npy'),allow_pickle=True)
        self.lick_times=self.lick_times[0]
        
        #load units
        self.units=pd.read_csv(os.path.join(path,'unit_table.csv'))
        self.units=self.units.set_index('id')
        self.good_units=self.units.query('quality == "good" and \
                        isi_viol < 0.5 and \
                        amplitude_cutoff < 0.1 and \
                        presence_ratio > 0.95')
        self.good_units=self.good_units.sort_values(by=['probe','peak_channel'])
        
        #load spike times
        self.spike_times=np.load(os.path.join(path,'spike_times_aligned.npy'),allow_pickle=True).item()
        
        #load frames
        self.frames=pd.read_csv(os.path.join(path,'frames_table.csv'))
        
        if len(glob.glob(os.path.join(path, 'rf_mapping*')))>0:
            #load RF mapping
            self.rf_trials=pd.read_csv(os.path.join(path,'rf_mapping_trials.csv'))
            
            #load RF frames
            self.rf_frames=pd.read_csv(os.path.join(path,'rf_mapping_frames.csv'))
        
        

# functions for binning the spiking data into a convenient shape for plotting
def makePSTH(spikes, startTimes, windowDur, binSize=0.001):
    '''
    Convenience function to compute a peri-stimulus-time histogram
    (see section 7.2.2 here: https://neuronaldynamics.epfl.ch/online/Ch7.S2.html)
    INPUTS:
        spikes: spike times in seconds for one unit
        startTimes: trial start times in seconds; the first spike count 
            bin will be aligned to these times
        windowDur: trial duration in seconds
        binSize: size of spike count bins in seconds
    OUTPUTS:
        Tuple of (PSTH, bins), where:
            PSTH gives the trial-averaged spike rate for 
                each time bin aligned to the start times;
            bins are the bin edges as defined by numpy histogram
    '''
    bins = np.arange(0,windowDur+binSize,binSize)
    counts = np.zeros(bins.size-1)
    for i,start in enumerate(startTimes):
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start+windowDur)
        counts = counts + np.histogram(spikes[startInd:endInd]-start, bins)[0]
    
    counts = counts/len(startTimes)
    return counts/binSize, bins


def make_neuron_time_trials_tensor(units, spike_times, stim_table, 
                                   time_before, trial_duration,
                                   bin_size=0.001):
    '''
    Function to make a tensor with dimensions [neurons, time bins, trials] to store
    the spike counts for stimulus presentation trials. 
    INPUTS:
        units: dataframe with unit info (same form as session.units table)
        stim_table: dataframe whose indices are trial ids and containing a
            'start_time' column indicating when each trial began
        time_before: seconds to take before each start_time in the stim_table
        trial_duration: total time in seconds to take for each trial
        bin_size: bin_size in seconds used to bin spike counts 
    OUTPUTS:
        unit_tensor: tensor storing spike counts. The value in [i,j,k] 
            is the spike count for neuron i at time bin j in the kth trial.
    '''
    neuron_number = len(units)
    trial_number = len(stim_table)
    unit_tensor = np.zeros((neuron_number, int(trial_duration/bin_size), trial_number))
    
    for u_counter, (iu, unit) in enumerate(units.iterrows()):
        unit_spike_times = spike_times[iu]
        for t_counter, (it, trial) in enumerate(stim_table.iterrows()):
            if 'stimStartTime' in trial:
                trial_start = trial.stimStartTime - time_before
            elif 'start_time' in trial:
                trial_start = trial.start_time - time_before
            unit_tensor[u_counter, :, t_counter] = makePSTH(unit_spike_times, 
                                                            [trial_start], 
                                                            trial_duration, 
                                                            binSize=bin_size)[0]
    return unit_tensor


# %%
def make_trial_da(good_units, spike_times, trials):
    
    #Make tensor
    time_before_flash = 1
    trial_duration = 3
    bin_size = 0.001
    trial_tensor = make_neuron_time_trials_tensor(good_units, spike_times, trials, 
                                                  time_before_flash, trial_duration, 
                                                  bin_size)
    
    # make xarray
    trial_da = xr.DataArray(trial_tensor, dims=("unit_id", "time", "trials"), 
                           coords={
                               "unit_id": good_units.index.values,
                               "time": np.arange(0, trial_duration, bin_size)-time_before_flash,
                               "trials": trials.index.values
                               })
    
    trial_tensor=None
    
    return trial_da

# %%
def plot_rasters(mainPath, trial_da, good_units, trials, templeton_rec):
    
    # initializing substrings
    sub1 = r"recordings"
    sub2 = r"processed"
     
    # getting index of substrings
    idx1 = mainPath.index(sub1)
    idx2 = mainPath.index(sub2)
     
    save_folder = ''
    
    # getting elements in between
    for idx in range(idx1 + len(sub1) + 1, idx2):
        save_folder = save_folder + mainPath[idx]
        
    
    if templeton_rec==True:
        save_folder_mainPath = r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\plots"
    else:
        save_folder_mainPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\plots"
    
    #clean this up
    save_folder_path = os.path.join(save_folder_mainPath,save_folder)
    
    if os.path.isdir(save_folder_path)==False:
            os.mkdir(save_folder_path)
            
    save_folder_path = os.path.join(save_folder_path,'rasters')
    
    if os.path.isdir(save_folder_path)==False:
            os.mkdir(save_folder_path)
    
    if templeton_rec == True:
        trials = trials.sort_values(by='trial_stim_dur',axis=0,ascending=True)
    
    for unit_id in good_units.index:
        
        probe_name=good_units['probe'].loc[unit_id]
        channel_num=str(good_units['peak_channel'].loc[unit_id])
        
        fig_name = 'Probe'+probe_name+'_unit'+str(unit_id)+'_ch'+channel_num+'_rasters'
        
        fig,ax=plt.subplots(2,2,figsize=(8,7))
        ax=ax.flatten()
        stim_types=['vis1','vis2','sound1','sound2']
        
        for si,ss in enumerate(stim_types):
            
            stim_trials = trials[trials['trialStimID']==ss].index.values
            sel_trials = trial_da.sel(trials=stim_trials)
            
            #find the ilocs of block transitions
            sel_trials_table = trials.loc[stim_trials]
            column_changes = sel_trials_table['trialstimRewarded'].shift() != sel_trials_table['trialstimRewarded']
            block_changes = sel_trials_table[column_changes]
            
            for it,tt in enumerate(sel_trials.trials.values):
                trial_spikes = sel_trials.sel(unit_id=unit_id,trials=tt)

                trial_spike_times = trial_spikes.time[trial_spikes.values.astype('bool')]
                
                ax[si].vlines(trials['trial_stim_dur'].loc[tt],ymin=it-.01,ymax=it+1.01,linewidth=1,color='tab:blue')
                ax[si].vlines(trial_spike_times,ymin=it,ymax=it+1,linewidth=0.75,color='k')
                
        
            if len(block_changes)>1:
                if block_changes.iloc[0]['trialstimRewarded']=='vis1':
                    start_block=1
                elif block_changes.iloc[0]['trialstimRewarded']=='sound1':
                    start_block=0
        
                for xx in np.asarray([0,2,4])+start_block:
                    start_iloc=sel_trials_table.index.get_loc(block_changes.index[xx])
                    if (xx+1)>(len(block_changes)-1):
                        end_iloc=len(sel_trials_table)
                    else:
                        end_iloc=sel_trials_table.index.get_loc(block_changes.index[xx+1])
                    temp_patch=patches.Rectangle([-0.5,start_iloc],1.5,end_iloc-start_iloc,
                                                color=[0.5,0.5,0.5],alpha=0.15)
                    ax[si].add_patch(temp_patch)
            
            if templeton_rec==True:
                ax[si].set_xlim([-0.5,1.5])
            else:
                ax[si].set_xlim([-0.5,1])
        
            ax[si].axvline(0,linewidth=1,color='tab:blue')
            
            ax[si].set_title(ss)
        
        fig.suptitle('unit:'+str(unit_id)+' Probe'+good_units['probe'].loc[unit_id]+' ch:'+str(good_units['peak_channel'].loc[unit_id]))
        
        fig.tight_layout()
        
        fig.savefig(os.path.join(save_folder_path,fig_name+'.png'), dpi=300, format=None, metadata=None,
                    bbox_inches=None, pad_inches=0.1,
                    facecolor='auto', edgecolor='auto',
                    backend=None,
                   )

        plt.close(fig)
    

# %%
def plot_heatmaps(mainPath,trial_da,trials,good_units,templeton_rec):    
    
    # initializing substrings
    sub1 = r"recordings" #pilot
    sub2 = r"processed"
     
    # getting index of substrings
    idx1 = mainPath.index(sub1)
    idx2 = mainPath.index(sub2)
     
    save_folder = ''
    
    # getting elements in between
    for idx in range(idx1 + len(sub1) + 1, idx2):
        save_folder = save_folder + mainPath[idx]
        
    
    if templeton_rec==True:
        save_folder_mainPath = r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\plots"
    else:
        save_folder_mainPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\plots"
    
    #clean this up
    save_folder_path = os.path.join(save_folder_mainPath,save_folder)
    
    if os.path.isdir(save_folder_path)==False:
            os.mkdir(save_folder_path)
            
    save_folder_path = os.path.join(save_folder_path,'heatmaps')
    
    if os.path.isdir(save_folder_path)==False:
            os.mkdir(save_folder_path)
    
    #find baseline mean and std per unit
    baseline_mean_per_trial=trial_da.sel(time=slice(-1,-0.9)).mean(dim=["time"])
    baseline_mean=baseline_mean_per_trial.mean(dim="trials").values
    baseline_std=baseline_mean_per_trial.std(dim="trials").values
    
    #1 - make average PSTHs across units for each stimulus
    
    # pre-allocate array [units x time x conditions]
    stimuli = np.unique(trials['trialStimID'])
    unit_frs_by_stim = np.zeros((len(trial_da.unit_id),len(trial_da.time),len(stimuli)))
    unit_frs_by_stim_visblock = np.zeros((len(trial_da.unit_id),len(trial_da.time),len(stimuli)))
    unit_frs_by_stim_audblock = np.zeros((len(trial_da.unit_id),len(trial_da.time),len(stimuli)))
    unit_frs_by_stim_resp = np.zeros((len(trial_da.unit_id),len(trial_da.time),len(stimuli)))
    unit_frs_by_stim_noresp = np.zeros((len(trial_da.unit_id),len(trial_da.time),len(stimuli)))
    
    # to start - responsiveness to each stimulus
    for ss,stim in enumerate(stimuli):
        # all trials
        stim_trials = (trials.query('trialStimID==@stim')).index.values
        unit_frs_by_stim[:,:,ss] = trial_da.sel(trials=stim_trials).mean(dim="trials").values
        unit_frs_by_stim[:,:,ss] = ((unit_frs_by_stim[:,:,ss].T- baseline_mean.T)/baseline_std.T).T
        #visual reward blocks
        stim_trials_visblock = (trials.query('trialStimID==@stim and trialstimRewarded == "vis1" ')).index.values
        unit_frs_by_stim_visblock[:,:,ss] = trial_da.sel(trials=stim_trials_visblock).mean(dim="trials").values
        unit_frs_by_stim_visblock[:,:,ss] = ((unit_frs_by_stim_visblock[:,:,ss].T- baseline_mean.T)/baseline_std.T).T
        #auditory rewarded blocks
        stim_trials_audblock = (trials.query('trialStimID==@stim and trialstimRewarded == "sound1" ')).index.values
        unit_frs_by_stim_audblock[:,:,ss] = trial_da.sel(trials=stim_trials_audblock).mean(dim="trials").values
        unit_frs_by_stim_audblock[:,:,ss] = ((unit_frs_by_stim_audblock[:,:,ss].T- baseline_mean.T)/baseline_std.T).T
        #response trials
        stim_trials_resp = (trials.query('trialStimID==@stim and trial_response == True')).index.values
        unit_frs_by_stim_resp[:,:,ss] = trial_da.sel(trials=stim_trials_resp).mean(dim="trials").values
        unit_frs_by_stim_resp[:,:,ss] = ((unit_frs_by_stim_resp[:,:,ss].T- baseline_mean.T)/baseline_std.T).T
        #nonresponse trials
        stim_trials_noresp = (trials.query('trialStimID==@stim and trial_response == False')).index.values
        unit_frs_by_stim_noresp[:,:,ss] = trial_da.sel(trials=stim_trials_noresp).mean(dim="trials").values
        unit_frs_by_stim_noresp[:,:,ss] = ((unit_frs_by_stim_noresp[:,:,ss].T- baseline_mean.T)/baseline_std.T).T


    # 2 - plot heatmaps    
    
    # probes borders for plotting
    probe_borders=np.where(good_units['probe'].iloc[:-1].values!=good_units['probe'].iloc[1:].values)[0]
    all_edges=np.hstack([0,probe_borders,len(good_units)])
    midpoints=all_edges[:-1]+(all_edges[1:]-all_edges[:-1])/2
    probe_labels=good_units['probe'].iloc[midpoints.astype('int')].values
    
    trial_avgs = [unit_frs_by_stim,
                  unit_frs_by_stim_visblock,
                  unit_frs_by_stim_audblock,
                  unit_frs_by_stim_resp,
                  unit_frs_by_stim_noresp]
    
    labels = ['all trials','visual blocks','auditory blocks','response trials','non response trials']
    
    for dd,data in enumerate(trial_avgs):
    
        fig,ax=plt.subplots(1,4,figsize=(10,8))
        for xx in range(1,5):
            im = ax[xx-1].imshow(data[:,:,xx],aspect='auto',vmin=-3,vmax=3,
                           cmap=plt.get_cmap('bwr'),interpolation='none',
                           extent=(-1,2,0,data.shape[0]))
    
            ax[xx-1].axvline(0,color='k',linestyle='--',linewidth=1)
            ax[xx-1].set_title(stimuli[xx])
            ax[xx-1].set_xlim(-0.5,1.5)
            ax[xx-1].hlines(data.shape[0]-probe_borders,xmin=-0.5,xmax=1.5,
                           color='k',linewidth=1)
            ax[xx-1].set_yticks(data.shape[0]-midpoints)
            ax[xx-1].set_yticklabels(probe_labels)
            if xx>1:
                ax[xx-1].set_yticklabels([])
        
        fig.suptitle(labels[dd])
        
        fig.subplots_adjust(bottom=0.1, right=0.8, top=0.9, hspace=0.3)
        cax = plt.axes([0.85, 0.1, 0.025, 0.8])
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.set_ylabel('z-scored firing rates')
        
        fig.savefig(os.path.join(save_folder_path,labels[dd]+'.png'), dpi=300, format=None, metadata=None,
                    bbox_inches=None, pad_inches=0.1,
                    facecolor='auto', edgecolor='auto',
                    backend=None,
                   )
    
        plt.close(fig)

# %%
def plot_rew_nonrew_rasters(mainPath, trial_da, good_units, spike_times, trials, lick_times, templeton_rec):
    
    # initializing substrings
    sub1 = r"recordings" #pilot
    sub2 = r"processed"
     
    # getting index of substrings
    idx1 = mainPath.index(sub1)
    idx2 = mainPath.index(sub2)
     
    save_folder = ''
    
    # getting elements in between
    for idx in range(idx1 + len(sub1) + 1, idx2):
        save_folder = save_folder + mainPath[idx]
        
    
    if templeton_rec==True:
        save_folder_mainPath = r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\plots"
    else:
        save_folder_mainPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\plots"
    
    #clean this up
    save_folder_path = os.path.join(save_folder_mainPath,save_folder)
    
    if os.path.isdir(save_folder_path)==False:
            os.mkdir(save_folder_path)
            
    save_folder_path = os.path.join(save_folder_path,'rew_nonrew')
    
    if os.path.isdir(save_folder_path)==False:
            os.mkdir(save_folder_path)
    
    # find first lick within ~1-2 seconds after the stimulus start
    first_lick_time=np.zeros(len(trials))
    first_lick_time[:]=np.nan
    for tt in range(0,len(trials)):
        if tt<len(trials)-1:
            first_lick=np.where((lick_times>trials['stimStartTime'].iloc[tt])&
                                (lick_times<trials['stimStartTime'].iloc[tt+1]))[0]
        else:
            first_lick=np.where((lick_times>trials['stimStartTime'].iloc[tt]))[0]
        
        if len(first_lick)>0:
            first_lick_time[tt]=lick_times[first_lick[0]]
         
    trials['first_lick_time']=first_lick_time
    trials['first_lick_latency']=trials['first_lick_time']-trials['stimStartTime']

    
    #find first licks after free rewards

    first_lick_rw=np.zeros(len(trials))
    first_lick_rw[:]=np.nan
    first_lick_post_rw=np.zeros(len(trials))
    first_lick_post_rw[:]=np.nan
    
    for tt in range(0,len(trials)):
        first_lick_rw_temp=np.where((lick_times>trials['stimStartTime'].iloc[tt]+0.1)&
                               (lick_times<trials['stimStartTime'].iloc[tt]+1))[0]
        if tt<len(trials)-1:
            first_lick_post_rw_temp=np.where((lick_times>trials['stimStartTime'].iloc[tt]+1)&
                                        (lick_times<trials['stimStartTime'].iloc[tt+1]))[0]
        else:
            first_lick_post_rw_temp=np.where((lick_times>trials['stimStartTime'].iloc[tt]+1))[0]
        if len(first_lick_rw_temp)>0:
            first_lick_rw[tt]=lick_times[first_lick_rw_temp[0]]
        if len(first_lick_post_rw_temp)>0:
            first_lick_post_rw[tt]=lick_times[first_lick_post_rw_temp[0]]
            
    trials['first_lick_rw']=first_lick_rw
    trials['first_lick_post_rw']=first_lick_post_rw
    
    
    lick_df=pd.DataFrame(lick_times,columns=['start_time'])
    
    first_lick_time = trials.query("trialStimID == trialstimRewarded and \
                                trial_rewarded == True")['first_lick_time'].values
    first_lick_df=pd.DataFrame(first_lick_time[~np.isnan(first_lick_time)],columns=['start_time'])
    
    #Make tensor
    time_before_flash = 1
    trial_duration = 3
    bin_size = 0.001
    lick_tensor = make_neuron_time_trials_tensor(good_units, spike_times, first_lick_df, 
                                                  time_before_flash, trial_duration, 
                                                  bin_size)
    
    # make xarray
    lick_da = xr.DataArray(lick_tensor, dims=("unit_id", "time", "licks"), 
                           coords={
                               "unit_id": good_units.index.values,
                               "time": np.arange(0, trial_duration, bin_size)-time_before_flash,
                               "licks": np.arange(0,len(first_lick_df))
                               })
    
    #free reward trials:
    free_rew_trials=trials.query("trialStimID == 'catch' and \
                                  trial_rewarded == True")
                                  
    free_rew_lick_df=pd.DataFrame(free_rew_trials['first_lick_post_rw']
                              [~np.isnan(free_rew_trials['first_lick_post_rw'])].values,
                              columns=['start_time'])

    free_rew_lick_tensor = make_neuron_time_trials_tensor(good_units, spike_times, free_rew_lick_df, 
                                                          time_before_flash, trial_duration, 
                                                          bin_size)
    
    # make xarray
    free_rew_lick_da = xr.DataArray(free_rew_lick_tensor, dims=("unit_id", "time", "licks"), 
                                   coords={
                                       "unit_id": good_units.index.values,
                                       "time": np.arange(0, trial_duration, bin_size)-time_before_flash,
                                       "licks": np.arange(0,len(free_rew_lick_df))
                                       })
    
    omitted_rew_trials = trials.query("trialStimID == trialstimRewarded and \
                                   trial_rewarded == False and \
                                   trial_response == True")
                                   
    omitted_rew_lick_df=pd.DataFrame(omitted_rew_trials['first_lick_rw']
                                  [~np.isnan(omitted_rew_trials['first_lick_rw'])].values,
                                  columns=['start_time'])
    
    omitted_rew_lick_tensor = make_neuron_time_trials_tensor(good_units, spike_times, omitted_rew_lick_df, 
                                                          time_before_flash, trial_duration, 
                                                          bin_size)
    omitted_rew_lick_da = xr.DataArray(omitted_rew_lick_tensor, dims=("unit_id", "time", "licks"), 
                                       coords={
                                           "unit_id": good_units.index.values,
                                           "time": np.arange(0, trial_duration, bin_size)-time_before_flash,
                                           "licks": np.arange(0,len(omitted_rew_lick_df))
                                           })
    
    #stim-aligned data arrays
    rewarded_trials = trials.query("trialStimID == trialstimRewarded and \
                                trial_rewarded == True")
    rewarded_trials = rewarded_trials.sort_values(by='first_lick_latency')
    
    omitted_rew_trials = trials.query("trialStimID == trialstimRewarded and \
                                       trial_rewarded == False")
    omitted_rew_trials = omitted_rew_trials.sort_values(by='first_lick_latency')
    
    rew_stim_tensor = make_neuron_time_trials_tensor(good_units, spike_times, rewarded_trials, 
                                                      time_before_flash, trial_duration, 
                                                      bin_size)
    rew_stim_da = xr.DataArray(rew_stim_tensor, dims=("unit_id", "time", "trial"), 
                               coords={
                                   "unit_id": good_units.index.values,
                                   "time": np.arange(0, trial_duration, bin_size)-time_before_flash,
                                   "trial": rewarded_trials.index.values
                                   })
    
    
    omitted_rew_stim_tensor = make_neuron_time_trials_tensor(good_units, spike_times, omitted_rew_trials, 
                                                              time_before_flash, trial_duration, 
                                                              bin_size)
    omitted_rew_stim_da = xr.DataArray(omitted_rew_stim_tensor, dims=("unit_id", "time", "trial"), 
                                       coords={
                                           "unit_id": good_units.index.values,
                                           "time": np.arange(0, trial_duration, bin_size)-time_before_flash,
                                           "trial": omitted_rew_trials.index.values
                                           })
        
    
    #plot lick-aligned rasters
    for unit_id in good_units.index:
        
        probe_name=good_units['probe'].loc[unit_id]
        channel_num=str(good_units['peak_channel'].loc[unit_id])
        
        fig_name = 'Probe'+probe_name+'_unit'+str(unit_id)+'_ch'+channel_num+'_lick_raster'
        
        fig,ax=plt.subplots(3,1)
        # rewarded licks
        ax[0].axvline(0)
        for tt in lick_da.licks.values:
            lick_spikes = lick_da.sel(unit_id=unit_id,licks=tt)
            lick_spike_times = lick_spikes.time[lick_spikes.values.astype('bool')]
            ax[0].vlines(lick_spike_times,ymin=tt,ymax=tt+1,linewidth=0.75,color='k')
        ax[0].set_xlim([lick_da.time[0],lick_da.time[-1]])
        ax[0].set_ylabel('rewarded')
        ax[0].set_xticklabels([])
        
        # omitted reward licks
        ax[1].axvline(0)
        for tt in omitted_rew_lick_da.licks.values:
            lick_spikes = omitted_rew_lick_da.sel(unit_id=unit_id,licks=tt)
            lick_spike_times = lick_spikes.time[lick_spikes.values.astype('bool')]
            ax[1].vlines(lick_spike_times,ymin=tt,ymax=tt+1,linewidth=0.75,color='k')
        ax[1].set_xlim([omitted_rew_lick_da.time[0],omitted_rew_lick_da.time[-1]])
        ax[1].set_ylabel('omitted rew')
        ax[1].set_xticklabels([])
        
        # free reward licks
        ax[2].axvline(0)
        for tt in free_rew_lick_da.licks.values:
            lick_spikes = free_rew_lick_da.sel(unit_id=unit_id,licks=tt)  
            lick_spike_times = lick_spikes.time[lick_spikes.values.astype('bool')]
            ax[2].vlines(lick_spike_times,ymin=tt,ymax=tt+1,linewidth=0.75,color='k')
        ax[2].set_xlim([free_rew_lick_da.time[0],free_rew_lick_da.time[-1]])
        ax[2].set_ylabel('free reward')
        ax[2].set_xlabel('time relative to lick (s)')
        
        fig.suptitle('unit:'+str(unit_id)+' Probe'+good_units['probe'].loc[unit_id]+' ch:'+str(good_units['peak_channel'].loc[unit_id]))
        
        fig.tight_layout()
    
        fig.savefig(os.path.join(save_folder_path,fig_name+'.png'), dpi=300, format=None, metadata=None,
                    bbox_inches=None, pad_inches=0.1,
                    facecolor='auto', edgecolor='auto',
                    backend=None,
                   )

        plt.close(fig)
    
    #plot lick-aligned rasters
    for unit_id in good_units.index:
        
        probe_name=good_units['probe'].loc[unit_id]
        channel_num=str(good_units['peak_channel'].loc[unit_id])
        
        fig_name = 'Probe'+probe_name+'_unit'+str(unit_id)+'_ch'+channel_num+'_stim_raster'
        
        fig,ax=plt.subplots(2,1)
        
        # rewarded licks
        ax[0].axvline(0)
        for it,tt in enumerate(rew_stim_da.trial.values):
            trial_spikes = rew_stim_da.sel(unit_id=unit_id,trial=tt)
            trial_spike_times = trial_spikes.time[trial_spikes.values.astype('bool')]
            ax[0].vlines(trial_spike_times,ymin=it,ymax=it+1,linewidth=0.75,color='k')
            ax[0].vlines(trials['first_lick_latency'].loc[tt],ymin=it,ymax=it+1,linewidth=2,color='tab:green')
        ax[0].set_xlim([rew_stim_da.time[0],rew_stim_da.time[-1]])
        ax[0].set_ylabel('rewarded')
        ax[0].set_xticklabels([])
        
        # omitted reward licks
        ax[1].axvline(0)
        for it,tt in enumerate(omitted_rew_stim_da.trial.values):
            trial_spikes = omitted_rew_stim_da.sel(unit_id=unit_id,trial=tt)
            trial_spike_times = trial_spikes.time[trial_spikes.values.astype('bool')]
            ax[1].vlines(trial_spike_times,ymin=it,ymax=it+1,linewidth=0.75,color='k')
            ax[1].vlines(trials['first_lick_latency'].loc[tt],ymin=it,ymax=it+1,linewidth=2,color='tab:green')
        ax[1].set_xlim([omitted_rew_stim_da.time[0],omitted_rew_stim_da.time[-1]])
        ax[1].set_ylabel('omitted rew')
        ax[1].set_xlabel('time relative to stim onset (s)')
        
        fig.suptitle('unit:'+str(unit_id)+' Probe'+good_units['probe'].loc[unit_id]
                     +' ch:'+str(good_units['peak_channel'].loc[unit_id]))
        
        fig.tight_layout()
    
        fig.savefig(os.path.join(save_folder_path,fig_name+'.png'), dpi=300, format=None, metadata=None,
                    bbox_inches=None, pad_inches=0.1,
                    facecolor='auto', edgecolor='auto',
                    backend=None,
                   )

        plt.close(fig)
        
    