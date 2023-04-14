# -*- coding: utf-8 -*-
"""
DR Analysis Utils

Useful functions for Dynamic Routing / Templeton analysis

Created on Mon Jan  9 11:16:14 2023

@author: ethan.mcbride
"""

import numpy as np
import scipy as sp
import pandas as pd
import xarray as xr
import scipy.signal as sg
import scipy.stats as st
import statsmodels.stats.multitest as stmulti
import matplotlib.pyplot as plt
from matplotlib import patches
import pickle
import os
import glob
import re

#Simple session class for analyzing DR data
class Session:
    '''
    Simple Session object for convenient analysis of ephys data
    INPUT:
        path: full path to data processed by DR_processing_script
    OUTPUT:
        session object containing all relevant session data
        session.metadata: metadata about the session
        session.trials: pandas DataFrame with information about each trial
        session.units: DataFrame with info about each unit
        session.good_units: DataFrame with info about units, filtered by quality metrics
        session.spike_times: dictionary of spike times, with each key corresponding to a unit_id in the units table
        session.lick_times: list of all recorded lick times during the session
        session.frames: table with time of each frame, along with running speed
        session.rf_trials: DataFrame with info about RF mapping trials
        session.rf_frames: table with time of each RF frame, along with running speed
    '''
    def __init__(self,params=None,path=None,mouseID=None,ephys_session_num=None):
        
        #load metadata
        if os.path.isfile(os.path.join(path,'metadata.pkl')):
            with open(os.path.join(path,'metadata.pkl'), 'rb') as handle:
                self.metadata = pickle.load(handle)
        elif mouseID is not None and ephys_session_num is not None:
            self.metadata={}
            self.metadata['mouseID']=mouseID
            self.metadata['ephys_session_num']=ephys_session_num
        else:
            print('metadata not found, input mouseID and ephys_session_num')
        
        #load trials
        self.trials=pd.read_csv(os.path.join(path,'trials_table.csv'),index_col=[0])
        trial_stim_dur=np.zeros(len(self.trials))
        for tt in range(0,len(self.trials)):
            if 'trial_sound_dur' in self.trials.columns:
                if self.trials['trial_sound_dur'].iloc[tt]>0:
                    trial_stim_dur[tt]=self.trials['trial_sound_dur'].iloc[tt]
            if 'trialSoundDur' in self.trials.columns:
                if self.trials['trialSoundDur'].iloc[tt]>0:
                    trial_stim_dur[tt]=self.trials['trialSoundDur'].iloc[tt]
            if 'trial_vis_stim_dur' in self.trials.columns:
                if self.trials['trial_vis_stim_dur'].iloc[tt]>0:
                    trial_stim_dur[tt]=self.trials['trial_vis_stim_dur'].iloc[tt]
            if 'trialVisStimFrames' in self.trials.columns:
                if self.trials['trialVisStimFrames'].iloc[tt]>0:
                    trial_stim_dur[tt]=self.trials['trialVisStimFrames'].iloc[tt]/60

        self.trials['trial_stim_dur']=trial_stim_dur
        
        #load lick times
        self.lick_times=np.load(os.path.join(path,'lick_times.npy'),allow_pickle=True)
        self.lick_times=self.lick_times[0]
        
        #load units
        self.units=pd.read_csv(os.path.join(path,'unit_table.csv'),index_col='id')
        if 'Unnamed: 0' in self.units.columns:
            self.units = self.units.drop(['Unnamed: 0'],axis='columns')

        self.good_units=self.units.query('quality == "good" and \
                                            isi_viol < 0.5 and \
                                            amplitude_cutoff < 0.1 and \
                                            presence_ratio > 0.95')
        self.good_units=self.good_units.sort_values(by=['probe','peak_channel'])
        
        #load spike times
        self.spike_times=np.load(os.path.join(path,'spike_times_aligned.npy'),allow_pickle=True).item()
        
        #load frames
        self.frames=pd.read_csv(os.path.join(path,'frames_table.csv'),index_col=[0])
        
        if len(glob.glob(os.path.join(path, 'rf_mapping*')))>0:
            #load RF mapping
            self.rf_trials=pd.read_csv(os.path.join(path,'rf_mapping_trials.csv'),index_col=[0])
            
            #load RF frames
            self.rf_frames=pd.read_csv(os.path.join(path,'rf_mapping_frames.csv'),index_col=[0])
        
        #compute average run speed for each trial
        avg_run_speed=np.zeros(len(self.trials))
        for tt in range(0,len(self.trials)):
            startFrame=self.trials['trialStimStartFrame'].iloc[tt]-66
            endFrame=self.trials['trialStimStartFrame'].iloc[tt]-6
            avg_run_speed[tt]=np.nanmean(self.frames['runningSpeed'][startFrame:endFrame])
        self.trials['avg_run_speed'] = avg_run_speed
        
        
    def assign_unit_areas(self):
        #check for area IDs

        tissuecyte_path = r"\\allen\programs\mindscope\workgroups\np-behavior\tissuecyte"
        self.units['area']=''
        self.good_units['area']=''
        self.units['AP_coord']=np.nan
        self.good_units['AP_coord']=np.nan
        self.units['DV_coord']=np.nan
        self.good_units['DV_coord']=np.nan
        self.units['ML_coord']=np.nan
        self.good_units['ML_coord']=np.nan
        
        if os.path.isdir(os.path.join(tissuecyte_path,self.metadata['mouseID'])):
            for probe in self.units['probe'].unique():
                if type(probe)==str:
                    channels_table_path=glob.glob(
                        os.path.join(tissuecyte_path,self.metadata['mouseID'],
                                     '*'+probe+str(self.metadata['ephys_session_num'])+'_channels*'))
                    if len(channels_table_path)==1:
                        channels_table=pd.read_csv(channels_table_path[0])
                        print('probe'+probe+' areas found')
                    else:
                        print('probe'+probe+' areas not found')
                        continue
                    for ic,chan in channels_table.iterrows():
                        chan_units = self.units.query('peak_channel == @chan.channel and \
                                                            probe == @probe').index
                        if len(chan_units)>0:
                            if 'region' in chan:
                                assign_area = chan['region']
                            elif 'channel_areas' in chan:
                                assign_area = chan['channel_areas']

                            self.units.loc[chan_units,'area'] = assign_area
                            
                            self.units.loc[chan_units,'AP_coord'] = chan['AP']
                            self.units.loc[chan_units,'DV_coord'] = chan['DV']
                            self.units.loc[chan_units,'ML_coord'] = chan['ML']
                        
                        chan_units = self.good_units.query('peak_channel == @chan.channel and \
                                                                    probe == @probe').index
                        if len(chan_units)>0:
                            if 'region' in chan:
                                assign_area = chan['region']
                            elif 'channel_areas' in chan:
                                assign_area = chan['channel_areas']
                            self.good_units.loc[chan_units,'area'] = assign_area
        
                            self.good_units.loc[chan_units,'AP_coord'] = chan['AP']
                            self.good_units.loc[chan_units,'DV_coord'] = chan['DV']
                            self.good_units.loc[chan_units,'ML_coord'] = chan['ML']
        
            self.units.loc[self.units['area'].isna(),'area']='N/A'
            self.good_units.loc[self.good_units['area'].isna(),'area']='N/A'
        
        else:
            print('tissuecyte folder not found')
            self.manual_assign_unit_areas()
            
        #shorten the area names to better lump together units
        #get rid of layers and/or sub-areas with dashes
        area_short = []
        for area in self.good_units['area']:
            if area=='N/A':
                short='N/A'
            elif area[:2]=='CA':
                short=area
            else:
                dig_ind=re.search(r"\d", area)
                dash_ind=re.search(r"-", area)
                if dig_ind!=None:
                    short=area[:dig_ind.start()]
                elif dash_ind!=None:
                    short=area[:dash_ind.start()]
                else:
                    short=area
                
            area_short.append(short)
            
        self.good_units['area_short']=area_short
            
            
        
            
            
    def manual_assign_unit_areas(self):
        area_assign_path = r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\estimate_brain_areas\estimate_brain_areas.csv"
        areas_table = pd.read_csv(area_assign_path)
        self.units['area']=''
        self.good_units['area']=''
        
        if np.any(areas_table['mouse']==int(self.metadata['mouseID'])):
            mouseID = int(self.metadata["mouseID"])
            session_num = self.metadata['ephys_session_num']
            areas_table=areas_table.query('mouse == @mouseID and session_num == @session_num').reset_index()
            
            for ss in range(0,len(areas_table)):
                probe=areas_table['probe'][ss]
                for chan in range(areas_table['start_chan'][ss],areas_table['end_chan'][ss]):
                    chan_units = self.units.query('peak_channel == @chan and \
                                                        probe == @probe').index
                    if len(chan_units)>0:
                        self.units.loc[chan_units,'area'] = areas_table['area'][ss]
                        
            for ss in range(0,len(areas_table)):
                probe=areas_table['probe'][ss]
                for chan in range(areas_table['start_chan'][ss],areas_table['end_chan'][ss]):
                    chan_units = self.good_units.query('peak_channel == @chan and \
                                                        probe == @probe').index
                    if len(chan_units)>0:
                        self.good_units.loc[chan_units,'area'] = areas_table['area'][ss]      
            print('found and loaded manual area assignments')
        else:
            print('manual area assignments not found')
            

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
def plot_rasters(mainPath, trial_da, good_units, trials, lick_times, templeton_rec):
    
    if templeton_rec==True:
        save_folder_mainPath = r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\plots"
        sub1 = r"recordings"
    else:
        save_folder_mainPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\plots"
        sub1 = r"Task 2 pilot"
 
    sub2 = r"processed"
    
    # getting index of substrings
    idx1 = mainPath.index(sub1)
    idx2 = mainPath.index(sub2)
     
    save_folder = ''
    
    # getting elements in between
    for idx in range(idx1 + len(sub1) + 1, idx2):
        save_folder = save_folder + mainPath[idx]
    
    #clean this up
    save_folder_path = os.path.join(save_folder_mainPath,save_folder)
    
    if os.path.isdir(save_folder_path)==False:
            os.mkdir(save_folder_path)
            
    save_folder_path = os.path.join(save_folder_path,'rasters_block_aligned')
    
    if os.path.isdir(save_folder_path)==False:
            os.mkdir(save_folder_path)
    
    if templeton_rec == True:
        trials = trials.sort_values(by='trial_stim_dur',axis=0,ascending=True)
    
    # find first licks after stimulus start
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
    
    #add green tick for correct response, red tick for incorrect response
    
    
    for unit_id in good_units.index:
        
        probe_name=good_units['probe'].loc[unit_id]
        channel_num=str(good_units['peak_channel'].loc[unit_id])
        
        # fig_name = 'Probe'+probe_name+'_unit'+str(unit_id)+'_ch'+channel_num+'_rasters'
        
        fig_name = 'unit'+str(unit_id)+'_'+good_units['area'].loc[unit_id]+'_probe'+probe_name
        fig_name = fig_name.replace("N/A","null")
        fig_name = fig_name.replace("/","-")
        
        fig,ax=plt.subplots(2,2,figsize=(10,9))
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
                ax[si].vlines(trial_spike_times,ymin=it,ymax=it+1,linewidth=0.65,color='k')
                
                if sel_trials_table['trialStimID'].loc[tt]==sel_trials_table['trialstimRewarded'].loc[tt]:
                    tick_color='g'
                else:
                    tick_color='r'
                if sel_trials_table['trial_response'].loc[tt]==True:
                    ax[si].vlines(sel_trials_table['first_lick_latency'].loc[tt],ymin=it,ymax=it+1,linewidth=2,color=tick_color)
        
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
        
        # fig.suptitle('unit:'+str(unit_id)+' Probe'+good_units['probe'].loc[unit_id]+' ch:'+str(good_units['peak_channel'].loc[unit_id]))
        
        fig.suptitle('unit:'+str(unit_id)+'  Probe'+good_units['probe'].loc[unit_id]+
                     '  ch:'+str(good_units['peak_channel'].loc[unit_id])+'  area:'+
                     good_units['area'].loc[unit_id])
        
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
    
    if templeton_rec==True:
        save_folder_mainPath = r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\plots"
        sub1 = r"recordings"
    else:
        save_folder_mainPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\plots"
        sub1 = r"Task 2 pilot"
 
    sub2 = r"processed"
    
    # getting index of substrings
    idx1 = mainPath.index(sub1)
    idx2 = mainPath.index(sub2)
     
    save_folder = ''
    
    # getting elements in between
    for idx in range(idx1 + len(sub1) + 1, idx2):
        save_folder = save_folder + mainPath[idx]
    
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
        
#%%
def plot_stim_vs_lick_aligned_rasters(session, mainPath, templeton_rec):
    
    if templeton_rec==True:
        save_folder_mainPath = r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\plots"
        sub1 = r"recordings"
    else:
        save_folder_mainPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\plots"
        sub1 = r"Task 2 pilot"
 
    sub2 = r"processed"
    
    # getting index of substrings
    idx1 = mainPath.index(sub1)
    idx2 = mainPath.index(sub2)
     
    save_folder = ''
    
    # getting elements in between
    for idx in range(idx1 + len(sub1) + 1, idx2):
        save_folder = save_folder + mainPath[idx]
    
    #clean this up
    save_folder_path = os.path.join(save_folder_mainPath,save_folder)
    
    if os.path.isdir(save_folder_path)==False:
            os.mkdir(save_folder_path)
            
    save_folder_path = os.path.join(save_folder_path,'rasters_stim_vs_lick_aligned')
    
    if os.path.isdir(save_folder_path)==False:
            os.mkdir(save_folder_path)
            
    # find first licks after stimulus start
    first_lick_time=np.zeros(len(session.trials))
    first_lick_time[:]=np.nan
    for tt in range(0,len(session.trials)):
        if tt<len(session.trials)-1:
            first_lick=np.where((session.lick_times>session.trials['stimStartTime'].iloc[tt])&
                                (session.lick_times<session.trials['stimStartTime'].iloc[tt+1]))[0]
        else:
            first_lick=np.where((session.lick_times>session.trials['stimStartTime'].iloc[tt]))[0]
        
        if len(first_lick)>0:
            first_lick_time[tt]=session.lick_times[first_lick[0]]        
    session.trials['first_lick_time']=first_lick_time
    session.trials['first_lick_latency']=session.trials['first_lick_time']-session.trials['stimStartTime']
    
    #split trials with response vs not
    response_trials = session.trials.query('trial_response == True')
    non_response_trials = session.trials.query('trial_response == False')
    
    #sort by stimulus, first lick latency
    response_trials_sorted = response_trials.sort_values(by=['trialStimID','first_lick_latency'])
    non_response_trials_sorted = non_response_trials.sort_values(by=['trialStimID'])
    
    
    #make a dataframe with 'first_lick_time' relabeled as 'start_time' so the tensor function likes it
    first_lick_df=response_trials['first_lick_time']
    first_lick_df=pd.DataFrame(first_lick_df)
    first_lick_df=first_lick_df.rename(columns={"first_lick_time": "start_time"})
    
    # loop through sessions and make unit xarray
    time_before_flash = 1.0
    trial_duration = 3
    bin_size = 0.001
    # Make tensor (3-D matrix [units,time,trials])
    trial_tensor = make_neuron_time_trials_tensor(session.good_units, session.spike_times, 
                                                  first_lick_df, time_before_flash, trial_duration, 
                                                  bin_size)
    # make xarray
    lick_aligned_da = xr.DataArray(trial_tensor, dims=("unit_id", "time", "trials"), 
                               coords={
                                   "unit_id": session.good_units.index.values,
                                   "time": np.arange(0, trial_duration, bin_size)-time_before_flash,
                                   "trials": first_lick_df.index.values
                                   })
        
    
    #plot stim-aligned rasters with lick time indicated
    for unit_id in session.good_units.index:
        
        fig_name = 'unit'+str(unit_id)+'_'+session.good_units['area'].loc[unit_id]+'_stim_aligned'
        fig_name = fig_name.replace("N/A","null")
        fig_name = fig_name.replace("/","-")
        
        fig,ax=plt.subplots(1,2,figsize=(10,7))
        ax=ax.flatten()
        
        # stim_types=['vis1','vis2','sound1','sound2','catch']
        stim_types=['catch','sound2','sound1','vis2','vis1',]
        
        for xx in range(0,2):
            ax[xx].axvline(0,linewidth=1)
            ax[xx].axvline(0.5,linewidth=1)
            
            trialcount_offset=0
            stim_trial_borders=[0]
        
            for si,ss in enumerate(stim_types):
                
                if xx==0:
                    stim_trials = response_trials_sorted.query('trialStimID == @ss')
                elif xx==1:
                    stim_trials = non_response_trials_sorted.query('trialStimID == @ss')
                    
                sel_trials = session.trial_da.sel(trials=stim_trials.index.values)
        
                for it,tt in enumerate(sel_trials.trials.values):
                    trial_spikes = sel_trials.sel(unit_id=unit_id,trials=tt)
                    trial_spike_times = trial_spikes.time[trial_spikes.values.astype('bool')]
                    ax[xx].vlines(trial_spike_times,ymin=it+trialcount_offset,ymax=it+1+trialcount_offset,linewidth=0.75,color='k')
        
                    if stim_trials['trialStimID'].loc[tt] == stim_trials['trialstimRewarded'].loc[tt]:
                        plot_color='g'
                    else:
                        plot_color='r'
                    
                    if xx==0:
                        ax[xx].vlines(stim_trials['first_lick_latency'].loc[tt],ymin=it-.01+trialcount_offset,
                                  ymax=it+1.01+trialcount_offset,linewidth=2,color=plot_color)
                ax[xx].axhline(trialcount_offset,color='k',linewidth=0.5)
                trialcount_offset=trialcount_offset+len(stim_trials)
                stim_trial_borders.append(trialcount_offset)
        
            stim_trial_borders=np.asarray(stim_trial_borders)
            stim_trial_midpoints=stim_trial_borders[:-1]+(stim_trial_borders[1:]-stim_trial_borders[:-1])/2
        
            for yy in np.asarray([1,3]):
                start_iloc=stim_trial_borders[yy]
                if (yy+1)>(len(stim_trial_borders)-1):
                    end_iloc=stim_trial_borders[-1]
                else:
                    end_iloc=stim_trial_borders[yy+1]
                    
                temp_patch=patches.Rectangle([-0.5,start_iloc],1.5,end_iloc-start_iloc,
                                            color=[0.5,0.5,0.5],alpha=0.10)
                ax[xx].add_patch(temp_patch)
        
            
            ax[xx].axhline(trialcount_offset,color='k',linewidth=0.5)      
            ax[xx].set_xlim([-0.25,1])
            ax[xx].set_ylim([0,trialcount_offset])
            ax[xx].set_yticks(stim_trial_midpoints)
            ax[xx].set_yticklabels(stim_types)
            ax[xx].set_xlabel('time (s)')
            
            if xx==0:
                ax[xx].set_title('reponse trials')
            elif xx==1:
                ax[xx].set_title('non-reponse trials')
                
                
        fig.suptitle('stim-aligned  unit:'+str(unit_id)+'  Probe'+session.good_units['probe'].loc[unit_id]+
                     '  ch:'+str(session.good_units['peak_channel'].loc[unit_id])+'  area:'+
                     session.good_units['area'].loc[unit_id])
        
        fig.tight_layout()
    
        fig.savefig(os.path.join(save_folder_path,fig_name+'.png'), dpi=300, format=None, metadata=None,
                    bbox_inches=None, pad_inches=0.1,
                    facecolor='auto', edgecolor='auto',
                    backend=None,
                   )

        plt.close(fig)
        
        
    
    
    #plot lick-aligned rasters with stim time indicated
    for unit_id in session.good_units.index:
        
        fig_name = 'unit'+str(unit_id)+'_'+session.good_units['area'].loc[unit_id]+'_lick_aligned'
        fig_name = fig_name.replace("N/A","null")
        fig_name = fig_name.replace("/","-")
        
        fig,ax=plt.subplots(1,1,figsize=(10,7))
        
        # stim_types=['vis1','vis2','sound1','sound2','catch']
        stim_types=['catch','sound2','sound1','vis2','vis1',]
        
        
        ax.axvline(0,linewidth=1)
        # ax.axvline(0.5,linewidth=1)
        
        trialcount_offset=0
        stim_trial_borders=[0]
        
        for si,ss in enumerate(stim_types):
        
            stim_trials = response_trials_sorted.query('trialStimID == @ss')
        
            sel_trials = lick_aligned_da.sel(trials=stim_trials.index.values)
        
            for it,tt in enumerate(sel_trials.trials.values):
                trial_spikes = sel_trials.sel(unit_id=unit_id,trials=tt)
                trial_spike_times = trial_spikes.time[trial_spikes.values.astype('bool')]
                ax.vlines(trial_spike_times,ymin=it+trialcount_offset,ymax=it+1+trialcount_offset,linewidth=0.75,color='k')
        
                if stim_trials['trialStimID'].loc[tt] == stim_trials['trialstimRewarded'].loc[tt]:
                    plot_color='g'
                else:
                    plot_color='r'
        
        
                ax.vlines(-stim_trials['first_lick_latency'].loc[tt],ymin=it-.01+trialcount_offset,
                          ymax=it+1.01+trialcount_offset,linewidth=2,color=plot_color)
                
            ax.axhline(trialcount_offset,color='k',linewidth=0.5)
            trialcount_offset=trialcount_offset+len(stim_trials)
            stim_trial_borders.append(trialcount_offset)
        
        stim_trial_borders=np.asarray(stim_trial_borders)
        stim_trial_midpoints=stim_trial_borders[:-1]+(stim_trial_borders[1:]-stim_trial_borders[:-1])/2
        
        for yy in np.asarray([1,3]):
            start_iloc=stim_trial_borders[yy]
            if (yy+1)>(len(stim_trial_borders)-1):
                end_iloc=stim_trial_borders[-1]
            else:
                end_iloc=stim_trial_borders[yy+1]
        
            temp_patch=patches.Rectangle([-1,start_iloc],2,end_iloc-start_iloc,
                                        color=[0.5,0.5,0.5],alpha=0.10)
            ax.add_patch(temp_patch)
        
        
        ax.axhline(trialcount_offset,color='k',linewidth=0.5)      
        ax.set_xlim([-1,1])
        ax.set_ylim([0,trialcount_offset])
        ax.set_yticks(stim_trial_midpoints)
        ax.set_yticklabels(stim_types)
        ax.set_xlabel('time (s)')
                
        fig.suptitle('lick-aligned  unit:'+str(unit_id)+'  Probe'+session.good_units['probe'].loc[unit_id]+
                     '  ch:'+str(session.good_units['peak_channel'].loc[unit_id])+'  area:'+
                     session.good_units['area'].loc[unit_id])
        
        fig.tight_layout()
        
        fig.savefig(os.path.join(save_folder_path,fig_name+'.png'), dpi=300, format=None, metadata=None,
                    bbox_inches=None, pad_inches=0.1,
                    facecolor='auto', edgecolor='auto',
                    backend=None,
                   )

        plt.close(fig)

# %%
def compute_smoothed_response_rate(session,trials_to_smooth=5):
    
    stims=session.trials['trialStimID'].unique()
    gwindow = sg.gaussian(trials_to_smooth*3, std=trials_to_smooth)
    
    for ss in stims:

        stimTrials=session.trials.query('trialStimID == @ss')
        stimTrials.loc[:,ss+'_smooth']=sg.convolve(stimTrials.loc[:,'trial_response'].values,gwindow,mode='same')/np.sum(gwindow)
        interp_func=sp.interpolate.interp1d(stimTrials.index,stimTrials[ss+'_smooth'])
    
        xnew=np.arange(np.min(stimTrials.index),np.max(stimTrials.index))
        temp_interp=interp_func(xnew)
        interp_full=np.zeros((len(session.trials)))
        interp_full[:]=np.nan
        interp_full[np.min(stimTrials.index):np.max(stimTrials.index)]=temp_interp
        session.trials[ss+'_interp']=interp_full
        
    #calculate smoothed cross- and intra-modal dprime
    cross_modal_dprime = np.zeros(len(session.trials))
    cross_modal_dprime[:] = np.nan
    intra_modal_dprime = np.zeros(len(session.trials))
    intra_modal_dprime[:] = np.nan
    
    for tt,trial in session.trials.iterrows():
        if (trial['vis_autoreward_trials']==True)|(trial['aud_autoreward_trials']==True):
            continue
            
        if trial['trialstimRewarded'] == 'vis1':
            temp_hit=trial['vis1_interp']
            temp_fa=trial['sound1_interp']
            temp_intra_fa=trial['vis2_interp']
            
        elif trial['trialstimRewarded'] == 'sound1':
            temp_hit=trial['sound1_interp']
            temp_fa=trial['vis1_interp']
            temp_intra_fa=trial['sound2_interp']
            
        if temp_hit==1:
            temp_hit=0.999
        elif temp_hit==0:
            temp_hit=0.001      
        if temp_fa==1:
            temp_fa=0.999
        elif temp_fa==0:
            temp_fa=0.001
        if temp_intra_fa==1:
            temp_intra_fa=0.999
        elif temp_intra_fa==0:
            temp_intra_fa=0.001
       
        cross_modal_dprime[tt]=(st.norm.ppf(temp_hit) - st.norm.ppf(temp_fa))
        intra_modal_dprime[tt]=(st.norm.ppf(temp_hit) - st.norm.ppf(temp_intra_fa))
        
    session.trials['cross_modal_dprime'] = cross_modal_dprime
    session.trials['intra_modal_dprime'] = intra_modal_dprime
    
        
    return session
    
# %% 
def plot_smoothed_response_rate(session, mainPath, templeton_rec, criteria=0.3):
    
    if templeton_rec==True:
        save_folder_mainPath = r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\plots"
        prefix = r"TempletonPilot"
    else:
        save_folder_mainPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\plots"
        prefix = r"DRpilot"
 
    session_folder=prefix+'_'+session.metadata['mouseID']+'_'+session.metadata['session_date']
          
    save_folder_path = os.path.join(save_folder_mainPath,session_folder,'behavior plots')
    
    if os.path.isdir(os.path.join(save_folder_mainPath,session_folder))==False:
            os.mkdir(os.path.join(save_folder_mainPath,session_folder))
    if os.path.isdir(save_folder_path)==False:
            os.mkdir(save_folder_path)
    
    # smooth & interpolate response rate to each stimulus across all trials
    trials_to_smooth=5
    gwindow = sg.gaussian(trials_to_smooth*3, std=trials_to_smooth)
    
    stims=session.trials['trialStimID'].unique()
    
    stim_list=np.asarray(['vis1','vis2','sound1','sound2','catch'])
    colors=np.asarray(['tab:blue','tab:green','tab:red','tab:orange','grey'])

    fig_name = 'smoothed_response_rate_'+session.metadata['mouseID']+'_rec'+str(session.metadata['ephys_session_num'])
    fig,ax=plt.subplots(1,1)
    
    for ss in stims:
        plot_color=colors[stim_list==ss][0]
        ax.plot(np.arange(0,len(session.trials)),session.trials[ss+'_interp'],color=plot_color)
    
        
    high_performance_trials=session.trials.query('abs(vis1_interp - sound1_interp)>=@criteria').index
    ax.plot(high_performance_trials,np.ones(len(high_performance_trials)),'k.')
    ax.plot(sg.convolve(session.trials['avg_run_speed']
                        /session.trials['avg_run_speed'].max(),
                        gwindow,mode='same')/np.sum(gwindow),'k',linewidth=0.5)
    
    ax.legend(stims)
    ax.set_title(session.metadata['mouseID']+' rec'+str(session.metadata['ephys_session_num']))
    ax.set_xlabel('trial number')
    ax.set_ylabel('smoothed response rate to stimulus')
    
    fig.tight_layout()
    
    fig.savefig(os.path.join(save_folder_path,fig_name+'.png'), dpi=300, format=None, metadata=None,
                bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None,
               )

    plt.close(fig)
    
    
    #plot smoothed dprime

    fig_name = 'smoothed_dprime_'+session.metadata['mouseID']+'_rec'+str(session.metadata['ephys_session_num'])
    fig,ax=plt.subplots(1,1)
    
    ax.plot(np.arange(0,len(session.trials)),session.trials['cross_modal_dprime'],'k',linewidth=1.5)
    ax.plot(np.arange(0,len(session.trials)),session.trials['intra_modal_dprime'],'k--',linewidth=1.5)       
    
    ax.axhline(2,color='tab:blue',linewidth=1)
    
    vis_autorewards=session.trials.query('vis_autoreward_trials == True').index.values
    aud_autorewards=session.trials.query('aud_autoreward_trials == True').index.values
    
    for xx in range(0,len(vis_autorewards)):
        ax.axvline(vis_autorewards[xx],color='b',alpha=0.5)
        ax.axvline(aud_autorewards[xx],color='r',alpha=0.5)
        if xx==0:
            ax.legend(['cross-modal dprime','intra-modal dprime','default threshold',
                      'vis autorewards','aud autorewards'])
            
    ax.set_title(session.metadata['mouseID']+' rec'+str(session.metadata['ephys_session_num']))
    ax.set_xlabel('trial number')
    ax.set_ylabel('smoothed dprime')
    
    fig.tight_layout()
    
    fig.savefig(os.path.join(save_folder_path,fig_name+'.png'), dpi=300, format=None, metadata=None,
                bbox_inches=None, pad_inches=0.1,
                facecolor='auto', edgecolor='auto',
                backend=None,
               )

    plt.close(fig)
    
# %%
def plot_area_PSTHs_by_block(session,templeton_rec,criteria=0.3):
    
    if templeton_rec==True:
        save_folder_mainPath = r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\plots"
        prefix = r"TempletonPilot"
    else:
        save_folder_mainPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\plots"
        prefix = r"DRpilot"
 
    session_folder=prefix+'_'+session.metadata['mouseID']+'_'+session.metadata['session_date']
          
    save_folder_path = os.path.join(save_folder_mainPath,session_folder,'PSTH_by_block')
    
    if os.path.isdir(os.path.join(save_folder_mainPath,session_folder))==False:
            os.mkdir(os.path.join(save_folder_mainPath,session_folder))
    if os.path.isdir(save_folder_path)==False:
            os.mkdir(save_folder_path)
    
    # plot PSTHs comparing vis-attend and aud-attend:
    attend_vis_trials=session.trials.query('(vis1_interp - sound1_interp)>=@criteria')
    attend_aud_trials=session.trials.query('(sound1_interp - vis1_interp)>=@criteria')
    
    # simple_areas = []
    # for aa in session.good_units['area'].unique():
    #     main_area = ''
    #     for char in aa:
    #         if char.isupper()|char.islower():
    #             main_area+=char
    #         else:
    #             break
    #     if main_area not in simple_areas:
    #         simple_areas.append(main_area)
    unique_areas=session.good_units['area'].unique()

    for aa in unique_areas:#simple_areas:
    
        #select units to plot by area
        area_sel=aa

        sel_units = session.good_units.query('area.str.contains(@area_sel)')
        
        fig_title=('area '+area_sel+' (n_units='+str(len(sel_units))+
               ')  vis_speed='+str(np.round(attend_vis_trials['avg_run_speed'].mean(),decimals=1))+
               '  aud_speed='+str(np.round(attend_aud_trials['avg_run_speed'].mean(),decimals=1)))
        fig_name=aa+'_PSTH_by_block_'+session.metadata['mouseID']+'_'+str(session.metadata['ephys_session_num'])
        fig_name=fig_name.replace('/','-')
        
        # average PSTH across selected units & each stimulus
        stimuli = np.unique(session.trials['trialStimID'])
        stim_PSTHs = {}
        
        for stim in stimuli:
            if 'trialOptoVoltage' in session.trials: 
                stim_trials_attend_vis = attend_vis_trials.query('trialStimID == @stim and trialOptoVoltage.isnull()').index
                stim_trials_attend_aud = attend_aud_trials.query('trialStimID == @stim and trialOptoVoltage.isnull()').index
            else:
                stim_trials_attend_vis = attend_vis_trials.query('trialStimID == @stim').index
                stim_trials_attend_aud = attend_aud_trials.query('trialStimID == @stim').index
            
            stim_PSTHs[stim]={}
            stim_PSTHs[stim]['attend_vis']=[]
            stim_PSTHs[stim]['attend_aud']=[]
            stim_PSTHs[stim]['attend_vis_n']=len(stim_trials_attend_vis)
            stim_PSTHs[stim]['attend_aud_n']=len(stim_trials_attend_aud)
            
            stim_PSTHs[stim]['attend_vis'].append(session.trial_da.sel(
                                                unit_id=sel_units.index,
                                                trials=stim_trials_attend_vis).mean(dim=['trials']))
            
            stim_PSTHs[stim]['attend_aud'].append(session.trial_da.sel(
                                                unit_id=sel_units.index,
                                                trials=stim_trials_attend_aud).mean(dim=['trials']))
        
        # smooth each unit's PSTH
        gwindow = sg.gaussian(25, std=10)
        stim_PSTH_smooth={}
        for stim in stimuli:
            stim_PSTH_smooth[stim]={}
            stim_PSTH_smooth[stim]['attend_vis']=np.zeros(stim_PSTHs[stim]['attend_vis'][0].shape)
            stim_PSTH_smooth[stim]['attend_aud']=np.zeros(stim_PSTHs[stim]['attend_aud'][0].shape)
            for iu,uu in enumerate(stim_PSTHs[stim]['attend_vis'][0].unit_id.values):
                stim_PSTH_smooth[stim]['attend_vis'][iu,:]=sg.convolve(stim_PSTHs[stim]['attend_vis'][0].sel(unit_id=uu),
                                                                        gwindow,mode='same')/np.sum(gwindow)
                stim_PSTH_smooth[stim]['attend_aud'][iu,:]=sg.convolve(stim_PSTHs[stim]['attend_aud'][0].sel(unit_id=uu),
                                                                        gwindow,mode='same')/np.sum(gwindow)
                
        #do the plotting
        fig,ax=plt.subplots(5,1,figsize=(6,12),sharex=True,sharey=True)

        trialconds=['attend_vis','attend_aud']
        
        for ss,stim in enumerate(stimuli):
            ax[ss].set_title(stim+'; n_vis='+str(stim_PSTHs[stim]['attend_vis_n'])+
                             '; n_aud='+str(stim_PSTHs[stim]['attend_aud_n']))
            
            for tt,trial_type in enumerate(trialconds):
                y=np.nanmean(stim_PSTH_smooth[stim][trial_type],0)
                err=np.nanstd(stim_PSTH_smooth[stim][trial_type],0)/np.sqrt(stim_PSTH_smooth[stim][trial_type].shape[0])
                linex=ax[ss].plot(stim_PSTHs[stim][trial_type][0].time, y)
                ax[ss].fill_between(stim_PSTHs[stim][trial_type][0].time, y-err, y+err,
                    alpha=0.2, edgecolor=None, facecolor=linex[0].get_color())
                
                if (ss==0)&(tt==1):
                    ax[ss].legend(trialconds)
                
                if tt==1:
                    ax[ss].axvline(0,color='k',linestyle='--',linewidth=0.75)
                    ax[ss].axvline(0.5,color='k',linestyle='--',linewidth=0.75)
                    ax[ss].set_ylabel('FR (Hz)')
                    
        ax[ss].set_xlabel('time (s)')
        
        # fig.tight_layout()
        
        fig.suptitle(fig_title)
        
        fig.savefig(os.path.join(save_folder_path,fig_name+'.png'), dpi=300, format=None, metadata=None,
                    bbox_inches=None, pad_inches=0.1,
                    facecolor='auto', edgecolor='auto',
                    backend=None,
                   )

        plt.close(fig)
        

# %%
def plot_area_PSTHs_by_response(session,templeton_rec,criteria=2):
    
    if templeton_rec==True:
        save_folder_mainPath = r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\plots"
        prefix = r"TempletonPilot"
    else:
        save_folder_mainPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\plots"
        prefix = r"DRpilot"
 
    session_folder=prefix+'_'+session.metadata['mouseID']+'_'+session.metadata['session_date']
          
    save_folder_path = os.path.join(save_folder_mainPath,session_folder,'PSTH_by_response')
    
    if os.path.isdir(os.path.join(save_folder_mainPath,session_folder))==False:
            os.mkdir(os.path.join(save_folder_mainPath,session_folder))
    if os.path.isdir(save_folder_path)==False:
            os.mkdir(save_folder_path)
    
    #only used for finding running speed currently
    attend_vis_trials=session.trials.query('(vis1_interp - sound1_interp)>=0.3')
    attend_aud_trials=session.trials.query('(sound1_interp - vis1_interp)>=0.3')

    unique_areas=session.good_units['area'].unique()
    unit_types=['RS','FS']

    for aa in unique_areas:#simple_areas:
        #select units to plot by area
        area_sel=aa
        
        for unit_type_sel in unit_types:
        
            if unit_type_sel=='FS':
                # sel_units = session.good_units.query('area.str.contains(@area_sel) and duration<=0.4')
                sel_units = session.good_units.query('area==@area_sel and duration<=0.4')
            elif unit_type_sel=='RS':
                # sel_units = session.good_units.query('area.str.contains(@area_sel) and duration>0.4')
                sel_units = session.good_units.query('area==@area_sel and duration>0.4')
                
            fig_title=('area '+area_sel+' '+unit_type_sel+' (n='+str(len(sel_units))+
                       ')  vis_speed='+str(np.round(attend_vis_trials['avg_run_speed'].mean(),decimals=1))+
                       '  aud_speed='+str(np.round(attend_aud_trials['avg_run_speed'].mean(),decimals=1))+
                       ' dprime>='+str(criteria))
            fig_name=aa+'_'+unit_type_sel+'_PSTH_by_response_'+session.metadata['mouseID']+'_'+str(session.metadata['ephys_session_num'])
            fig_name=fig_name.replace('/','-')
            
            # average PSTH across selected units & each stimulus
            stimuli = np.unique(session.trials['trialStimID'])
            response_types = ['hit','miss','fa','cr']
            stim_PSTHs = {}
            stim_PSTHs_z = {}
            
            # compute baseline mean and std
            sel_baseline_trials=session.trials.query('cross_modal_dprime >= 1.5 and intra_modal_dprime >= 1.5').index
            baseline = session.trial_da.sel(unit_id=sel_units.index,
                                             time=slice(-0.5,0),
                                             trials=sel_baseline_trials).mean(dim=['time'])
            baseline_mean = baseline.mean(dim=['trials'])
            baseline_std = baseline.std(dim=['trials'])
            
            for stim in stimuli:
                if 'trialOptoVoltage' in session.trials: 
                    stim_trials = session.trials.query('trialStimID == @stim and \
                                                        cross_modal_dprime >= 1.5 and\
                                                        intra_modal_dprime >= 1.5 and\
                                                        trialOptoVoltage.isnull()')
                else:
                    stim_trials = session.trials.query('trialStimID == @stim and \
                                                        cross_modal_dprime >= 1.5 and\
                                                        intra_modal_dprime >= 1.5')
                
                stim_response_trials = {}
                stim_response_trials['hit'] = stim_trials.query('trial_response == True and trialStimID == trialstimRewarded')
                stim_response_trials['miss'] = stim_trials.query('trial_response == False and trialStimID == trialstimRewarded')
                stim_response_trials['fa'] = stim_trials.query('trial_response == True and trialStimID != trialstimRewarded')
                stim_response_trials['cr'] = stim_trials.query('trial_response == False and trialStimID != trialstimRewarded')
                stim_PSTHs[stim]={}
                stim_PSTHs_z[stim]={}
                
                for rr in response_types:
                    stim_PSTHs[stim][rr+'_n']=[]
                    stim_PSTHs[stim][rr+'_n']=len(stim_response_trials[rr])
                    stim_PSTHs[stim][rr]=[]
                    stim_PSTHs[stim][rr].append(session.trial_da.sel(
                                                        unit_id=sel_units.index,
                                                        trials=stim_response_trials[rr].index).mean(dim=['trials']))
                    
                    stim_PSTHs_z[stim][rr+'_n']=[]
                    stim_PSTHs_z[stim][rr+'_n']=len(stim_response_trials[rr])
                    stim_PSTHs_z[stim][rr]=[]
                    stim_PSTHs_z[stim][rr].append((session.trial_da.sel(
                                                  unit_id=sel_units.index,
                                                  trials=stim_response_trials[rr].index).mean(dim=['trials'])-
                                                  baseline_mean)/baseline_std)
                    
            # smooth each unit's PSTH
            gwindow = sg.gaussian(20, std=5)
            stim_PSTH_smooth={}
            stim_PSTH_z_smooth={}
            for stim in stimuli:
                stim_PSTH_smooth[stim]={}
                stim_PSTH_z_smooth[stim]={}
                
                for rr in response_types:
                    stim_PSTH_smooth[stim][rr]=np.zeros(stim_PSTHs[stim][rr][0].shape)
                    stim_PSTH_z_smooth[stim][rr]=np.zeros(stim_PSTHs[stim][rr][0].shape)
                    for iu,uu in enumerate(stim_PSTHs[stim][rr][0].unit_id.values):
                        stim_PSTH_smooth[stim][rr][iu,:]=sg.convolve(stim_PSTHs[stim][rr][0].sel(unit_id=uu),
                                                                                gwindow,mode='same')/np.sum(gwindow)
                        stim_PSTH_z_smooth[stim][rr][iu,:]=sg.convolve(stim_PSTHs_z[stim][rr][0].sel(unit_id=uu),
                                                                                gwindow,mode='same')/np.sum(gwindow)

            fig,ax=plt.subplots(5,1,figsize=(6,12),sharex=True,sharey=True)

            response_types = ['miss','cr','fa','hit']
            color_sel = ['tab:blue','k','tab:red','tab:green']
            
            for ss,stim in enumerate(stimuli):
                ax[ss].set_title(stim+'; n_hit='+str(stim_PSTHs[stim]['hit_n'])+
                                 '; n_miss='+str(stim_PSTHs[stim]['miss_n'])+
                                 '; n_fa='+str(stim_PSTHs[stim]['fa_n'])+
                                 '; n_cr='+str(stim_PSTHs[stim]['cr_n']))
                line_all=[]
                for tt,trial_type in enumerate(response_types):
                    y=np.nanmean(stim_PSTH_smooth[stim][trial_type],0)
                    err=np.nanstd(stim_PSTH_smooth[stim][trial_type],0)/np.sqrt(stim_PSTH_smooth[stim][trial_type].shape[0])
                    linex=ax[ss].plot(stim_PSTHs[stim][trial_type][0].time, y, color=color_sel[tt])
                    line_all.append(linex[0])
                    ax[ss].fill_between(stim_PSTHs[stim][trial_type][0].time, y-err, y+err,
                        alpha=0.2, edgecolor=None, facecolor=linex[0].get_color())
                
                if ss==0:      
                    ax[ss].legend(line_all,response_types)
                ax[ss].axvline(0,color='k',linestyle='--',linewidth=0.75)
                ax[ss].axvline(0.5,color='k',linestyle='--',linewidth=0.75)
                ax[ss].set_ylabel('FR (Hz)')
                ax[ss].set_xlim([-0.25,0.75])
            ax[ss].set_xlabel('time (s)')
            
            fig.suptitle(fig_title)
            
            # fig.tight_layout()
            
            fig.savefig(os.path.join(save_folder_path,fig_name+'.png'), dpi=300, format=None, metadata=None,
                        bbox_inches=None, pad_inches=0.1,
                        facecolor='auto', edgecolor='auto',
                        backend=None,
                       )

            plt.close(fig)
            
            fig_name=aa+'_'+unit_type_sel+'_PSTH_by_response_zscore_'+session.metadata['mouseID']+'_'+str(session.metadata['ephys_session_num'])
            fig_name=fig_name.replace('/','-')
            fig,ax=plt.subplots(5,1,figsize=(6,12),sharex=True,sharey=True)

            response_types = ['miss','cr','fa','hit']
            color_sel = ['tab:blue','k','tab:red','tab:green']
            
            for ss,stim in enumerate(stimuli):
                ax[ss].set_title(stim+'; n_hit='+str(stim_PSTHs[stim]['hit_n'])+
                                 '; n_miss='+str(stim_PSTHs[stim]['miss_n'])+
                                 '; n_fa='+str(stim_PSTHs[stim]['fa_n'])+
                                 '; n_cr='+str(stim_PSTHs[stim]['cr_n']))
                line_all=[]
                for tt,trial_type in enumerate(response_types):
                    y=np.nanmean(stim_PSTH_z_smooth[stim][trial_type],0)
                    err=np.nanstd(stim_PSTH_z_smooth[stim][trial_type],0)/np.sqrt(stim_PSTH_z_smooth[stim][trial_type].shape[0])
                    linex=ax[ss].plot(stim_PSTHs_z[stim][trial_type][0].time, y, color=color_sel[tt])
                    line_all.append(linex[0])
                    ax[ss].fill_between(stim_PSTHs_z[stim][trial_type][0].time, y-err, y+err,
                        alpha=0.2, edgecolor=None, facecolor=linex[0].get_color())
                
                if ss==0:      
                    ax[ss].legend(line_all,response_types)
                ax[ss].axvline(0,color='k',linestyle='--',linewidth=0.75)
                ax[ss].axvline(0.5,color='k',linestyle='--',linewidth=0.75)
                ax[ss].set_ylabel('z-scored FR')
                ax[ss].set_xlim([-0.25,0.75])
            ax[ss].set_xlabel('time (s)')
            
            fig.suptitle(fig_title)
            
            # fig.tight_layout()
            
            fig.savefig(os.path.join(save_folder_path,fig_name+'.png'), dpi=300, format=None, metadata=None,
                        bbox_inches=None, pad_inches=0.1,
                        facecolor='auto', edgecolor='auto',
                        backend=None,
                       )

            plt.close(fig)

# %%
def compute_stimulus_response(session):
    
    # define time window (0-200ms?)
    prestim_win = [-0.2,0.0]
    stim_win = [0.0,0.2]
    
    # sum spikes in these time windows
    pre_stim_response_da = session.trial_da.sel(time=slice(prestim_win[0],prestim_win[1])).mean(dim='time')
    stim_response_da = session.trial_da.sel(time=slice(stim_win[0],stim_win[1])).mean(dim='time')
    
    # # record the mean change (and sign)
    # stim_evoked=stim_response_da-pre_stim_response_da
    
    n_units=len(session.good_units)
    
    stim_response_p={}
    stim_response_p_adj={}
    stim_response_avg={}
    stimuli = session.trials['trialStimID'].unique()
    # stimuli = ['vis1','vis2','sound1','sound2','catch']
    for stim in stimuli:
        stim_response_p[stim+'_sig']=np.zeros((n_units))
        stim_response_p[stim+'_sig'][:]=np.nan
        stim_response_p_adj[stim+'_sig']=np.zeros((n_units))
        stim_response_p_adj[stim+'_sig'][:]=np.nan
        stim_response_avg[stim+'_resp']=np.zeros((n_units))
        stim_response_avg[stim+'_resp'][:]=np.nan
        
    all_pvals=[]
    all_stim_ind=[]
    #loop through stimuli
    for stim_ind,stim in enumerate(stimuli):
        stim_trials = session.trials.query('trialStimID == @stim and \
                                            aud_autoreward_trials == False and \
                                            vis_autoreward_trials == False').index
        
        for ui,unit_sel in enumerate(session.good_units.index):
            
            # statistical test pre vs. stim FR
            pre_stim_sel_unit = pre_stim_response_da.sel(unit_id=unit_sel,trials=stim_trials)
            stim_sel_unit = stim_response_da.sel(unit_id=unit_sel,trials=stim_trials)
            
            if np.all((stim_sel_unit.values-pre_stim_sel_unit.values)!=0):
                h,stim_response_p[stim+'_sig'][ui]=st.wilcoxon(pre_stim_sel_unit.values,stim_sel_unit.values)
            else:
                stim_response_p[stim+'_sig'][ui]=1
                
            stim_response_avg[stim+'_resp'][ui]=(np.nanmean(stim_sel_unit.values-pre_stim_sel_unit.values)/
                                         np.nanstd(pre_stim_sel_unit.values))
            
        all_pvals.append(stim_response_p[stim])
        all_stim_ind.append(np.ones(len(stim_response_p[stim]))*stim_ind)
        
    ########
    # keep track of how many tests, do multiple comparisons corrections
    all_pvals=np.hstack(all_pvals)
    all_stim_ind=np.hstack(all_stim_ind)
    hyp_test,adj_pvals=stmulti.fdrcorrection(all_pvals,alpha=0.05,method='indep')  
    # translate back to lists for each stimulus
    
    for stim_ind,stim in enumerate(stimuli):
        stim_response_p_adj[stim+'_sig']=adj_pvals[all_stim_ind==stim_ind]
        
        
    unit_selectivity = pd.DataFrame(stim_response_p_adj,columns=['vis1_sig','vis2_sig','sound1_sig','sound2_sig','catch_sig'],
                                  index=session.good_units.index)<0.05
    
    unit_response = pd.DataFrame(stim_response_avg,
                             columns=['vis1_resp','vis2_resp','sound1_resp','sound2_resp','catch_resp'],
                             index=session.good_units.index)
    
    session.good_units = pd.concat([session.good_units,unit_selectivity,unit_response],axis=1)
    
    return session


# %%
def compute_block_modulation(session):
    
    # define time window (0-100ms?)
    prestim_win = [-0.2,0.0]
    stim_win = [0.0,0.2]
    
    # sum spikes in these time windows
    pre_stim_response_da = session.trial_da.sel(time=slice(prestim_win[0],prestim_win[1])).mean(dim='time')
    stim_response_da = session.trial_da.sel(time=slice(stim_win[0],stim_win[1])).mean(dim='time')
    
    n_units=len(session.good_units)
    
    time_windows = ['block_diff_pre_sig','block_diff_stim_sig']

    pre_block_diff_p=np.zeros((n_units))
    pre_block_diff_p[:]=np.nan
    pre_block_diff_avg=np.zeros((n_units))
    pre_block_diff_avg[:]=np.nan
    
    stim_block_diff_p=np.zeros((n_units))
    stim_block_diff_p[:]=np.nan
    stim_block_diff_avg=np.zeros((n_units))
    stim_block_diff_avg[:]=np.nan
    
    all_pvals=[]
    all_ind=[]
    
    vis_block_trials = session.trials.query('trialstimRewarded == "vis1" and \
                                         aud_autoreward_trials == False and \
                                         vis_autoreward_trials == False and \
                                         cross_modal_dprime >= 1.5').index
    
    aud_block_trials = session.trials.query('trialstimRewarded == "sound1" and \
                                         aud_autoreward_trials == False and \
                                         vis_autoreward_trials == False and \
                                         cross_modal_dprime >= 1.5').index
    
    for ui,unit_sel in enumerate(session.good_units.index):
    
        # statistical test pre vs. stim FR
        pre_stim_sel_unit_vis = pre_stim_response_da.sel(unit_id=unit_sel,trials=vis_block_trials)
        pre_stim_sel_unit_aud = pre_stim_response_da.sel(unit_id=unit_sel,trials=aud_block_trials)
        pre_stim_all = pre_stim_response_da.sel(unit_id=unit_sel)
        
        stim_sel_unit_vis = stim_response_da.sel(unit_id=unit_sel,trials=vis_block_trials)
        stim_sel_unit_aud = stim_response_da.sel(unit_id=unit_sel,trials=aud_block_trials)
        stim_all = stim_response_da.sel(unit_id=unit_sel)
        

        h,pre_block_diff_p[ui]=st.ranksums(pre_stim_sel_unit_vis.values,pre_stim_sel_unit_aud.values)

        pre_block_diff_avg[ui]=((np.nanmean(pre_stim_sel_unit_vis.values)-np.nanmean(pre_stim_sel_unit_aud.values))/
                                      np.nanstd(pre_stim_all.values))
        
        h,stim_block_diff_p[ui]=st.ranksums(stim_sel_unit_vis.values,stim_sel_unit_aud.values)

        stim_block_diff_avg[ui]=((np.nanmean(stim_sel_unit_vis.values)-np.nanmean(stim_sel_unit_aud.values))/
                                      np.nanstd(stim_all.values))
        
    
    all_pvals.append(pre_block_diff_p)
    all_ind.append(np.ones(len(pre_block_diff_p))*0)
    
    all_pvals.append(stim_block_diff_p)
    all_ind.append(np.ones(len(stim_block_diff_p))*1)
    
    
    ########
    # keep track of how many tests, do multiple comparisons corrections
    all_pvals=np.hstack(all_pvals)
    all_ind=np.hstack(all_ind)
    hyp_test,adj_pvals=stmulti.fdrcorrection(all_pvals,alpha=0.05,method='indep')  
    
    # translate back to lists for each stimulus
    block_diff_p_adj={}
    for t_ind,t_wind in enumerate(time_windows):
        block_diff_p_adj[t_wind]=adj_pvals[all_ind==t_ind]
        
    block_selectivity=pd.DataFrame(block_diff_p_adj,columns=['block_diff_pre_sig','block_diff_stim_sig'],
                                   index=session.good_units.index)<0.05
    
    block_diffs = pd.DataFrame({'block_diff_pre':pre_block_diff_avg,'block_diff_stim':stim_block_diff_avg},
                               index=session.good_units.index)
    
    session.good_units = pd.concat([session.good_units,block_selectivity,block_diffs],axis=1)
    
    return session

# %%
def plot_block_modulated_PSTHs(session,templeton_rec):
    
    if templeton_rec==True:
        save_folder_mainPath = r"\\allen\programs\mindscope\workgroups\templeton\TTOC\pilot recordings\plots"
        prefix = r"TempletonPilot"
    else:
        save_folder_mainPath = r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\plots"
        prefix = r"DRpilot"
 
    session_folder=prefix+'_'+session.metadata['mouseID']+'_'+session.metadata['session_date']
          
    save_folder_path = os.path.join(save_folder_mainPath,session_folder,'block_modulated_PSTHs')
    
    if os.path.isdir(os.path.join(save_folder_mainPath,session_folder))==False:
            os.mkdir(os.path.join(save_folder_mainPath,session_folder))
    if os.path.isdir(save_folder_path)==False:
            os.mkdir(save_folder_path)
    
    unique_areas=session.good_units['area_short'].unique()
    
    gwindow = sg.gaussian(25, std=10)
 
    vis_block_trials = session.trials.query('trialstimRewarded == "vis1" and \
                                         aud_autoreward_trials == False and \
                                         vis_autoreward_trials == False and \
                                         cross_modal_dprime >= 1.5')
    
    aud_block_trials = session.trials.query('trialstimRewarded == "sound1" and \
                                         aud_autoreward_trials == False and \
                                         vis_autoreward_trials == False and \
                                         cross_modal_dprime >= 1.5')
    
    plot_stimuli=['vis1','vis2','sound1','sound2','catch']
    
    #loop through areas, plot avg PSTHs for each
    for aa in unique_areas:
        
        block_mod_increased_units=session.good_units.query('block_diff_pre_sig == True and block_diff_pre>0 and \
                                                            area_short == @aa')
        block_mod_decreased_units=session.good_units.query('block_diff_pre_sig == True and block_diff_pre<0 and \
                                                            area_short == @aa')
        non_block_mod_units=session.good_units.query('block_diff_pre_sig == False and area_short == @aa')
        
        
        
        fig,ax=plt.subplots(3,5,sharex=True,sharey=True,figsize=(15,8))
        plot_colors=['b','r']
        for ss,stim in enumerate(plot_stimuli):
            for tr,trial_sel in enumerate([vis_block_trials,aud_block_trials]):
                
                if 'trialOptoVoltage' in session.trials.columns:
                    trial_stim_sel = trial_sel.query('trialOptoVoltage.isnull() and trialStimID==@stim').index
                else:
                    trial_stim_sel = trial_sel.query('trialStimID==@stim').index
    
                block_mod_increased_da=session.trial_da.sel(unit_id=block_mod_increased_units.index,
                                                            trials=trial_stim_sel).mean(dim=['trials','unit_id'])
    
                block_mod_decreased_da=session.trial_da.sel(unit_id=block_mod_decreased_units.index,
                                                            trials=trial_stim_sel).mean(dim=['trials','unit_id'])
    
                non_block_mod_da=session.trial_da.sel(unit_id=non_block_mod_units.index,
                                                      trials=trial_stim_sel).mean(dim=['trials','unit_id'])
    
                block_mod_increased_smoothed=sg.convolve(block_mod_increased_da.values,gwindow,mode='same')/np.sum(gwindow)
                block_mod_decreased_smoothed=sg.convolve(block_mod_decreased_da.values,gwindow,mode='same')/np.sum(gwindow)
                non_block_mod_smoothed=sg.convolve(non_block_mod_da.values,gwindow,mode='same')/np.sum(gwindow)
    
                ax[0,ss].plot(non_block_mod_da.time,non_block_mod_smoothed,color=plot_colors[tr])
                ax[1,ss].plot(block_mod_decreased_da.time,block_mod_decreased_smoothed,color=plot_colors[tr])
                ax[2,ss].plot(block_mod_increased_da.time,block_mod_increased_smoothed,color=plot_colors[tr])
            
            for xx in range(0,3):
                ax[xx,ss].set_xlim([-0.5,1.0])
                ax[xx,ss].axvline(0,color='k',linestyle='--',linewidth=0.75)
                ax[xx,ss].axvline(0.5,color='k',linestyle='--',linewidth=0.75)
                if ss==0:
                    ax[xx,ss].set_ylabel('FR (Hz)')
            
            ax[0,ss].legend(['vis block','aud block'])
            ax[0,ss].set_title(stim+'; '+'non block mod n='+str(len(non_block_mod_units.index)))
            ax[1,ss].set_title(stim+'; '+'aud block preferring n='+str(len(block_mod_decreased_units.index)))
            ax[2,ss].set_title(stim+'; '+'vis block preferring n='+str(len(block_mod_increased_units.index)))
            ax[2,ss].set_xlabel('time rel to stimulus (s)')
        
        fig.suptitle('area: '+aa+' '+session.metadata['mouseID']+' '+str(session.metadata['ephys_session_num']))
        fig.tight_layout()
        
        fig_name=aa+'_block_modulated_PSTHs_'+session.metadata['mouseID']+'_'+str(session.metadata['ephys_session_num'])
        fig_name=fig_name.replace('/','-')
        
        fig.savefig(os.path.join(save_folder_path,fig_name+'.png'), dpi=300, format=None, metadata=None,
                    bbox_inches=None, pad_inches=0.1,
                    facecolor='auto', edgecolor='auto',
                    backend=None,
                   )

        plt.close(fig)