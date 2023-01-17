# -*- coding: utf-8 -*-
"""
DR Processing Utilities:
    
Functions for processing ephys recordings for the DR and Templeton projects

Created on Fri Jan 13 15:02:53 2023

@author: ethan.mcbride
"""

import numpy as np
import pandas as pd
import h5py
import os
import ast
import sync.sync as sync
import probeSync
import ecephys
import glob

# %%
def load_behavior_data(behavPath):
    # behavior/stimuli

    f = h5py.File(behavPath,'r')
    
    trialEndFrame = f['trialEndFrame'][:]
    nTrials = trialEndFrame.size
    trialStartFrame = f['trialStartFrame'][:nTrials]
    startTime = f['startTime'][()]
    
    #vis params
    trialVisStimFrames = f['trialVisStimFrames'][:nTrials]
    
    #sound params
    trialSoundDur = f['trialSoundDur'][:nTrials]
    trialSoundArray = f['trialSoundArray'][:nTrials]
    trialSoundFreq = f['trialSoundFreq'][:nTrials]
    soundSampleRate = f['soundSampleRate'][()]

    #behavior stuff
    lickFrames=f['lickFrames'][:]
    rewardFrames=f['rewardFrames'][:]
    trialAutoRewarded=f['trialAutoRewarded'][:]
    responseWindow=f['responseWindow'][:]

    blockStim=f['blockStim'][:].astype('str')
    blockStimRewarded=f['blockStimRewarded'][:].astype('str')
    trialBlock=f['trialBlock'][:nTrials]
    trialStimID=f['trialStim'][:nTrials].astype('str')
    trialBlockRewarded=blockStimRewarded[trialBlock-1]
    trialStimStartFrame=f['trialStimStartFrame'][:nTrials]
    trialEndFrame=f['trialEndFrame'][:nTrials]
    
    opto_vars=0
    if 'trialGalvoVoltage' in list(f.keys()):
        opto_vars=1
        trialGalvoVoltage=f['trialGalvoVoltage'][:nTrials]
        trialOptoDur=f['trialOptoDur'][:nTrials]
        trialOptoOnsetFrame=f['trialOptoOnsetFrame'][:nTrials]
        trialOptoVoltage=f['trialOptoVoltage'][:nTrials]
    
    if len(f['rotaryEncoderCount'][:])>0:
        wheelPos=(f['rotaryEncoderCount'][:]/f['rotaryEncoderCountsPerRev'][()])*(2*np.pi*f['wheelRadius'][()])*60
        deltaWheelPos=np.zeros((len(wheelPos)))
        deltaWheelPos[1:]=wheelPos[1:]-wheelPos[:-1]
    else:
        deltaWheelPos=[]
    
    if type(blockStim)==str:
        blockStim=ast.literal_eval(blockStim)
    for xx in range(0,len(blockStim)):
        if type(blockStim[xx])==str:
            blockStim[xx]=ast.literal_eval(blockStim[xx])
    
    f.close()
    
    #re-generate trial sound arrays if I forgot to save them (only early pilot recordings)
    # assumes sound1=6kHz and sound2=10kHz
    if len(trialSoundArray)==0:
        soundHanningDur = 0.005 # seconds
    
        trialSoundArray=[]
    
        for it,tt in enumerate(trialSoundDur):
            t = np.arange(0,tt,1/soundSampleRate)
            soundArray = np.sin(2 * np.pi * trialSoundFreq[it,0] * t)
    
            if len(soundArray)>0:
                hanningSamples = int(soundSampleRate * soundHanningDur)
                hanningWindow = np.hanning(2 * hanningSamples + 1)
                soundArray[:hanningSamples] *= hanningWindow[:hanningSamples]
                soundArray[-hanningSamples:] *= hanningWindow[hanningSamples+1:]
    
            trialSoundArray.append(soundArray)
    
    
    #define hit/miss/fa/cr/autoreward trials
    unique_blocks=np.unique(trialBlock)
    
    #specify vis go/nogo, aud go/nogo
    vis_go_trials=np.zeros((nTrials)).astype('bool')
    vis_nogo_trials=np.zeros((nTrials)).astype('bool')
    vis_hit_trials=np.zeros((nTrials)).astype('bool')
    vis_false_alarm_trials=np.zeros((nTrials)).astype('bool')
    vis_miss_trials=np.zeros((nTrials)).astype('bool')
    vis_correct_reject_trials=np.zeros((nTrials)).astype('bool')
    vis_autoreward_trials=np.zeros((nTrials)).astype('bool')
    # vis_manualreward_trials=np.zeros((nTrials)).astype('bool')
    
    aud_go_trials=np.zeros((nTrials)).astype('bool')
    aud_nogo_trials=np.zeros((nTrials)).astype('bool')
    aud_hit_trials=np.zeros((nTrials)).astype('bool')
    aud_false_alarm_trials=np.zeros((nTrials)).astype('bool')
    aud_miss_trials=np.zeros((nTrials)).astype('bool')
    aud_correct_reject_trials=np.zeros((nTrials)).astype('bool')
    aud_autoreward_trials=np.zeros((nTrials)).astype('bool')
    # aud_manualreward_trials=np.zeros((nTrials)).astype('bool')
    
    catch_trials=np.zeros((nTrials)).astype('bool')
    catch_resp_trials=np.zeros((nTrials)).astype('bool')
    
    trial_response=np.zeros((nTrials)).astype('bool')
    trial_rewarded=np.zeros((nTrials)).astype('bool')
    
    block=[]
    stim_rewarded=[]
    
    for bb in range(0,len(unique_blocks)):
        
        blockTrialStart=np.where(trialBlock==unique_blocks[bb])[0][0]
        blockTrialEnd=np.where(trialBlock==unique_blocks[bb])[0][-1]+1
        
        for tt in range(blockTrialStart,blockTrialEnd):
            block.append(bb)
            stim_rewarded.append(blockStimRewarded[bb])
            
            if (tt>=len(trialEndFrame)):
                break
            temp_start_frame=trialStimStartFrame[tt]-250 #preStimFramesFixed
            temp_end_frame=trialEndFrame[tt]
    
            temp_licks=[]
            
            if len(lickFrames)>0:
                temp_licks=np.copy(lickFrames)
                temp_licks=temp_licks[(temp_licks>temp_start_frame)&(temp_licks<temp_end_frame)]-trialStimStartFrame[tt]
    
            if len(rewardFrames)>0:
                temp_reward=np.copy(rewardFrames)
                temp_reward=temp_reward[(temp_reward>temp_start_frame)&(temp_reward<temp_end_frame)]-trialStimStartFrame[tt]

            temp_RW_lick=0
            for ii in temp_licks:
                if (ii>=responseWindow[0])&(ii<=responseWindow[1]):
                    temp_RW_lick=1
                    trial_response[tt]=True
                    
            for rr in temp_reward:
                if (rr>=responseWindow[0]):
                    trial_rewarded[tt]=True
    
            #visual-go block
            if (trialStimID[tt] == 'vis1'):
                vis_go_trials[tt]=True
                if ~trialAutoRewarded[tt]:
                    if temp_RW_lick:
                        vis_hit_trials[tt]=True
                    else:
                        vis_miss_trials[tt]=True
                else:
                    vis_autoreward_trials[tt]=True
    
    
            elif (trialStimID[tt] == 'vis2'):
                vis_nogo_trials[tt]=True
                if temp_RW_lick:
                    vis_false_alarm_trials[tt]=True
                else:
                    vis_correct_reject_trials[tt]=True
    
    
            elif trialStimID[tt] == 'catch':
                catch_trials[tt]=True
                if temp_RW_lick:
                    catch_resp_trials[tt]==True
    
            elif ('sound1' in trialStimID[tt]):
                aud_go_trials[tt]=True
                if ~trialAutoRewarded[tt]:
                    if temp_RW_lick:
                        aud_hit_trials[tt]=True
                    else:
                        aud_miss_trials[tt]=True
    
            elif ('sound2' in trialStimID[tt]):
                aud_nogo_trials[tt]=True
                if ~trialAutoRewarded[tt]:
                    if temp_RW_lick:
                        aud_false_alarm_trials[tt]=True
                    else:
                        aud_correct_reject_trials[tt]=True
                else:
                    aud_autoreward_trials[tt]=True
    
    
    trials_dict={
        'trialStartFrame':trialStartFrame,
        'trialStimID':trialStimID,
        'trialstimRewarded':stim_rewarded,
        'trial_response':trial_response,
        'trial_rewarded':trial_rewarded,
        'trial_sound_dur':trialSoundDur,
        'trial_vis_stim_dur':trialVisStimFrames/60,
        
        'vis_go_trials':vis_go_trials,
        'vis_nogo_trials':vis_nogo_trials,
        'vis_hit_trials':vis_hit_trials,
        'vis_false_alarm_trials':vis_false_alarm_trials,
        'vis_miss_trials':vis_miss_trials,
        'vis_correct_reject_trials':vis_correct_reject_trials,
        'vis_autoreward_trials':vis_autoreward_trials,
    
        'aud_go_trials':aud_go_trials,
        'aud_nogo_trials':aud_nogo_trials,
        'aud_hit_trials':aud_hit_trials,
        'aud_false_alarm_trials':aud_false_alarm_trials,
        'aud_miss_trials':aud_miss_trials,
        'aud_correct_reject_trials':aud_correct_reject_trials,
        'aud_autoreward_trials':aud_autoreward_trials,
        
        'catch_trials':catch_trials,
        'catch_resp_trials':catch_resp_trials,
        
        'trialStimStartFrame':trialStimStartFrame,
    }
    
    trials_df=pd.DataFrame.from_dict(trials_dict)
    
    if opto_vars:
        trials_df['trialGalvoVoltage_x']=trialGalvoVoltage[:,0]
        trials_df['trialGalvoVoltage_y']=trialGalvoVoltage[:,1]
        trials_df['trialOptoDur']=trialOptoDur
        trials_df['trialOptoOnsetFrame']=trialOptoOnsetFrame
        trials_df['trialOptoVoltage']=trialOptoVoltage

    return trials_df, trialSoundArray, trialSoundDur, soundSampleRate, deltaWheelPos, startTime

# %%
def load_rf_mapping(rfPath):
    
    #load RF mapping stimulus info
    f = h5py.File(rfPath,'r')
    
    rf_stimFrames=f['stimFrames'][()]
    rf_soundDur=f['soundDur'][()]
    rf_trialSoundArray=f['trialSoundArray'][:]
    
    rf_dict={
        'trialsX': f['trialVisXY'][:,0],
        'trialsY': f['trialVisXY'][:,1],
        'trialGratingOri': f['trialGratingOri'][:],
        'stimStartFrame': f['stimStartFrame'][:],
    }
    
    aud_vect=np.zeros(len(f['trialGratingOri'][:]))
    vis_vect=np.zeros(len(f['trialGratingOri'][:]))
    
    if 'trialAMNoiseFreq' in list(f.keys()):
        rf_dict['trialAMNoiseFreq']=f['trialAMNoiseFreq'][:]
        aud_vect[~np.isnan(rf_dict['trialAMNoiseFreq'])]=1
        
    if 'trialToneFreq' in list(f.keys()):
        rf_dict['trialToneFreq']=f['trialToneFreq'][:]
        aud_vect[~np.isnan(rf_dict['trialToneFreq'])]=1
        
    if 'trialSoundFreq' in list(f.keys()):
        rf_dict['trialToneFreq']=f['trialSoundFreq'][:]
        aud_vect[~np.isnan(rf_dict['trialToneFreq'])]=1
        
    if 'trialFullFieldContrast' in list(f.keys()):
        rf_dict['trialFullFieldContrast']=f['trialFullFieldContrast'][:]
        vis_vect[~np.isnan(rf_dict['trialFullFieldContrast'])]=1
    
    vis_vect[~np.isnan(rf_dict['trialGratingOri'])]=1
        
    rf_df=pd.DataFrame.from_dict(rf_dict)
    
    trialStimType=[]
    for rr in range(0,len(rf_df)):
       if aud_vect[rr]==1:
           trialStimType.append('sound')
       elif vis_vect[rr]==1:
           trialStimType.append('vis')
       else:
           trialStimType.append('NaN')
        
    trialStimType=np.asarray(trialStimType)
    
    rf_df['trialStimType']=trialStimType
            
    if len(f['rotaryEncoderCount'][:])>0:
        rf_wheelPos=(f['rotaryEncoderCount'][:]/f['rotaryEncoderCountsPerRev'][()])*(2*np.pi*f['wheelRadius'][()])*60
        rf_deltaWheelPos=np.zeros((len(rf_wheelPos)))
        rf_deltaWheelPos[1:]=rf_wheelPos[1:]-rf_wheelPos[:-1]
        
    f.close()
    
    return rf_df, rf_trialSoundArray, rf_soundDur, rf_deltaWheelPos

# %%
def load_sound_pilot_data(behavPath):
    
    # behavior/stimuli

    f = h5py.File(behavPath,'r')
    
    trialEndFrame = f['trialEndFrame'][:]
    nTrials = trialEndFrame.size
    trialStartFrame = f['trialStartFrame'][:nTrials]
    trialStimStartFrame = f['trialStimStartFrame'][:nTrials]
    trialStimID=f['trialStim'][:nTrials].astype('str')
    startTime = f['startTime'][()]
    
    #vis params
    trialVisStimFrames = f['trialVisStimFrames'][:nTrials]
    trialGratingOri = f['trialGratingOri'][:nTrials]
    trialGratingPhase = f['trialGratingPhase'][:nTrials]
    trialVisStimContrast = f['trialVisStimContrast'][:nTrials]
    
    #sound params
    soundSampleRate = f['soundSampleRate'][()]
    
    trialSoundAM = f['trialSoundAM'][:nTrials]
    trialSoundDur = f['trialSoundDur'][:nTrials]
    trialSoundArray = f['trialSoundArray'][:nTrials]
    trialSoundFreq = f['trialSoundFreq'][:nTrials]
    trialSoundType = f['trialSoundType'][:nTrials].astype('str')
    trialSoundVolume = f['trialSoundVolume'][:nTrials]
    
    if len(f['rotaryEncoderCount'][:])>0:
        wheelPos=(f['rotaryEncoderCount'][:]/f['rotaryEncoderCountsPerRev'][()])*(2*np.pi*f['wheelRadius'][()])*60
        deltaWheelPos=np.zeros((len(wheelPos)))
        deltaWheelPos[1:]=wheelPos[1:]-wheelPos[:-1]
        
    
    trials_dict={
        'trialStartFrame':trialStartFrame,
        'trialEndFrame':trialEndFrame,
        'trialStimID':trialStimID,
        'trialStimStartFrame':trialStimStartFrame,
        
        'trialVisStimFrames':trialVisStimFrames,
        'trialGratingOri':trialGratingOri,
        'trialGratingPhase':trialGratingPhase[:,0],
        'trialVisStimContrast':trialVisStimContrast,
        
        'trialSoundAM':trialSoundAM,
        'trialSoundDur':trialSoundDur,
        'trialSoundFreq_0':trialSoundFreq[:,0],
        'trialSoundFreq_1':trialSoundFreq[:,1],
        'trialSoundType':trialSoundType,
        'trialSoundVolume':trialSoundVolume,
    }
    
    trials_df=pd.DataFrame.from_dict(trials_dict)
        
    f.close()
    
    
    return trials_df, trialSoundArray, trialSoundDur, soundSampleRate, deltaWheelPos, startTime

# %%
def sync_data_streams(syncPath,ephysPath):
    
    # get sync data
    syncDataset = sync.Dataset(syncPath)
    syncBarcodeRising,syncBarcodeFalling = probeSync.get_sync_line_data(syncDataset,'barcode_ephys')
    syncBarcodeTimes,syncBarcodes = ecephys.extract_barcodes_from_times(syncBarcodeRising,syncBarcodeFalling)
    
    # find probe & nidaq directory names
    probeDirNames = [pr for pr in os.listdir(os.path.join(ephysPath,'continuous')) if '-AP' in pr]
    probeNames = [pr[-4] for pr in probeDirNames]
    nidaqDirName = [pr for pr in os.listdir(os.path.join(ephysPath,'continuous')) if 'NI-DAQmx' in pr]
    
    syncData = {key: {'dirName': dirName} for key,dirName in zip(probeNames+['nidaq'],probeDirNames+nidaqDirName)}
    
    # find probe and nidaq offsets and sample rates
    ephysSampleRate = 30000
    for key,d in syncData.items():
        datTimestampsPath = os.path.join(ephysPath,'continuous',d['dirName'],'sample_numbers.npy')
        ttlStatesPath = os.path.join(ephysPath,'events',d['dirName'],'TTL','states.npy')
        ttlTimestampsPath = os.path.join(os.path.dirname(ttlStatesPath),'sample_numbers.npy')
        
        datTimestamps = np.load(datTimestampsPath)/ephysSampleRate
        
        ttlStates = np.load(ttlStatesPath)
        ttlTimestamps = np.load(ttlTimestampsPath)/ephysSampleRate - datTimestamps[0]
        
        barcodeRising = ttlTimestamps[ttlStates>0]
        barcodeFalling = ttlTimestamps[ttlStates<0]
        barcodeTimes,barcodes = ecephys.extract_barcodes_from_times(barcodeRising,barcodeFalling)
        
        shift,sampleRate,endpoints = ecephys.get_probe_time_offset(syncBarcodeTimes,syncBarcodes,barcodeTimes,barcodes,0,ephysSampleRate)
    
        syncData[key]['shift'] = shift
        syncData[key]['sampleRate'] = sampleRate
        if (np.isnan(sampleRate))|(~np.isfinite(sampleRate)):
            syncData[key]['sampleRate'] = ephysSampleRate

    return syncData, probeNames, probeDirNames

# %%
def sync_data_streams_sound_pilot(syncPath,ephysPath):
    
    syncDataset = sync.Dataset(syncPath)

    #load barcodes
    syncBarcodeRising,syncBarcodeFalling = probeSync.get_sync_line_data(syncDataset,'barcode_ephys')
    syncBarcodeTimes,syncBarcodes = ecephys.extract_barcodes_from_times(syncBarcodeRising,syncBarcodeFalling)
    
    # find probe & nidaq directory names
    probeDirNames = [pr for pr in os.listdir(os.path.join(ephysPath,'continuous')) if '-AP' in pr]
    probeNames = [pr[-4] for pr in probeDirNames]
    nidaqDirName = [pr for pr in os.listdir(os.path.join(ephysPath,'continuous')) if 'NI-DAQmx' in pr]
    
    syncData = {key: {'dirName': dirName} for key,dirName in zip(probeNames+['nidaq'],probeDirNames+nidaqDirName)}
    
    
    ephysSampleRate = 30000
    for key,d in syncData.items():
        datTimestampsPath = os.path.join(ephysPath,'continuous',d['dirName'],'timestamps.npy')#'timestamps.npy')
        ttlStatesPath = os.path.join(ephysPath,'events',d['dirName'],'TTL','channel_states.npy')
        ttlTimestampsPath = os.path.join(os.path.dirname(ttlStatesPath),'timestamps.npy')#'timestamps.npy')
    
        datTimestamps = np.load(datTimestampsPath)/ephysSampleRate
        
        ttlStates = np.load(ttlStatesPath)
        ttlTimestamps = np.load(ttlTimestampsPath)/ephysSampleRate - datTimestamps[0]
        
        barcodeRising = ttlTimestamps[ttlStates>0]
        barcodeFalling = ttlTimestamps[ttlStates<0]
        barcodeTimes,barcodes = ecephys.extract_barcodes_from_times(barcodeRising,barcodeFalling)
        
        shift,sampleRate,endpoints = ecephys.get_probe_time_offset(syncBarcodeTimes,syncBarcodes,barcodeTimes,barcodes,0,ephysSampleRate)
    
        syncData[key]['shift'] = shift
        syncData[key]['sampleRate'] = sampleRate
        if (np.isnan(sampleRate))|(~np.isfinite(sampleRate)):
            syncData[key]['sampleRate'] = ephysSampleRate
    
    return syncData, probeNames, probeDirNames

# %%
def align_trial_times(trials_df, syncData, syncPath, ephysPath, trialSoundArray, 
                      trialSoundDur, probeNames, probeDirNames, soundSampleRate,
                      deltaWheelPos, RF_first):
    
    syncDataset = sync.Dataset(syncPath)
    
    ### check for the 'stimulus_on' sync line!
    
    #load vsyncs
    vsyncRising,vsyncFalling = probeSync.get_sync_line_data(syncDataset,'vsync_stim')
    vsyncTimes = vsyncFalling[1:] if vsyncFalling[0] < vsyncRising[0] else vsyncFalling
    #load phtodiode
    photodiodeRising,photodiodeFalling = probeSync.get_sync_line_data(syncDataset,'stim_photodiode')
    photodiodeAll=np.sort(np.hstack([photodiodeRising[photodiodeRising>vsyncTimes[0]],
                                     photodiodeFalling[photodiodeFalling>vsyncTimes[0]]]))
    # find break in vsyncs to isolate the initial, behavior block
    stimBreak = np.where(np.diff(vsyncFalling)>30)[0]
    
    # get stim running signal (not always present)
    stimRunningRising,stimRunningFalling = probeSync.get_sync_line_data(syncDataset,'stim_running')
    if (len(stimRunningRising)>0)&(len(stimRunningFalling)>0):
        stimRunningFalling = stimRunningFalling[1:] if stimRunningFalling[0]<stimRunningRising[0] else stimRunningFalling
    
    if len(stimBreak)==0:
        stimBreak = len(vsyncFalling)-1
    else:
        stimBreak = stimBreak[0]
        
    if RF_first:
        vsyncBehavior = vsyncTimes[(vsyncTimes>stimRunningRising[1])&
                                   (vsyncTimes<=stimRunningFalling[1])]
        photodiodeBehavior = photodiodeAll[(photodiodeAll>vsyncBehavior[0])]
        photodiodeBehavior = photodiodeBehavior[:len(vsyncBehavior)]
    else:
        if len(stimRunningRising)>0:
            vsyncBehavior = vsyncTimes[(vsyncTimes>stimRunningRising[0])&
                                       (vsyncTimes<=stimRunningFalling[0])]
            photodiodeBehavior = photodiodeAll[(photodiodeAll>vsyncBehavior[0])]
        else:
            photodiodeBehavior = photodiodeAll[:stimBreak]
            vsyncBehavior = vsyncTimes[:stimBreak]
            
        photodiodeBehavior = photodiodeBehavior[:len(vsyncBehavior)]
    
    # average white & black photodiode transitions and apply to the last 2 trials?
    frameDelayAvg = np.zeros((len(photodiodeBehavior)))
    for ff in range(1,len(photodiodeBehavior),2):
        frameDelayAvg[ff-1:ff+1] = np.mean(photodiodeBehavior[ff-1:ff+1]-vsyncBehavior[ff-1:ff+1])
        
    nTrials = len(trials_df)
    trialStim = trials_df['trialStimID'].values
    stimStartFrame = trials_df['trialStimStartFrame']
    
    #find stimulus latencies
    nidaqDatPath = os.path.join(ephysPath,'continuous',syncData['nidaq']['dirName'],'continuous.dat')
    
    numAnalogCh = 8
    nidaqData = np.memmap(nidaqDatPath,dtype='int16',mode='r')    
    nidaqData = np.reshape(nidaqData,(int(nidaqData.size/numAnalogCh),-1)).T
    
    speakerCh = 1
    microphoneCh = 3
    microphoneData = nidaqData[microphoneCh]
    
    stimLatency = np.zeros(nTrials)
    stimStartTime = np.zeros(nTrials)
    preTime = 0.15
    postTime = 0.15
    for trial,stim in enumerate(trialStim.astype('str')):
        startFrame = stimStartFrame[trial]
        startTime = vsyncBehavior[startFrame] #+ syncData['nidaq']['shift']
        if 'sound' in stim:
            soundStartTime = startTime + syncData['nidaq']['shift']
            stimDur = trialSoundDur[trial]
            startSample = int((soundStartTime - preTime) * syncData['nidaq']['sampleRate'])
            endSample = int((soundStartTime + stimDur + postTime) * syncData['nidaq']['sampleRate'])
            t = np.arange(endSample-startSample) / syncData['nidaq']['sampleRate'] - preTime
            sound = trialSoundArray[trial]
            tInterp = np.arange(-preTime,preTime+stimDur+postTime,1/soundSampleRate)
            mic = microphoneData[startSample:endSample]
            micInterp = np.interp(tInterp,t,mic)
            c = np.correlate(micInterp,sound,'valid')
            stimLatency[trial] = tInterp[np.argmax(c)]
            stimStartTime[trial] = tInterp[np.argmax(c)]+startTime
            
        elif 'vis' in stim:
            stimLatency[trial] = frameDelayAvg[trial] # approximately
            stimStartTime[trial] = frameDelayAvg[trial]+startTime
            
        else:
            stimStartTime[trial] = startTime
            
    trials_df['stimStartTime'] = stimStartTime
    trials_df['stimLatency'] = stimLatency
    
    if 'Unnamed: 0' in trials_df.columns:
        trials_df = trials_df.drop(['Unnamed: 0'],axis='columns')
    
    if len(deltaWheelPos)>0:
        frames = {
            'vsyncTimes':vsyncBehavior,
            'frameDelay':frameDelayAvg,
            'runningSpeed':deltaWheelPos[:len(vsyncBehavior)]
            }
    else:
        frames = {
            'vsyncTimes':vsyncBehavior,
            'frameDelay':frameDelayAvg,
            }
    
    frames_df=pd.DataFrame.from_dict(frames)
    
    return trials_df, frames_df

# %%
def align_rf_trial_times(rf_df, syncData, syncPath, ephysPath, rf_trialSoundArray, 
                         rf_soundDur, soundSampleRate, rf_deltaWheelPos, RF_first):
    
    syncDataset = sync.Dataset(syncPath)
    
    #load vsyncs
    vsyncRising,vsyncFalling = probeSync.get_sync_line_data(syncDataset,'vsync_stim')
    vsyncTimes = vsyncFalling[1:] if vsyncFalling[0] < vsyncRising[0] else vsyncFalling
    #load phtodiode
    photodiodeRising,photodiodeFalling = probeSync.get_sync_line_data(syncDataset,'stim_photodiode')
    photodiodeAll=np.sort(np.hstack([photodiodeRising[photodiodeRising>vsyncTimes[0]],
                                     photodiodeFalling[photodiodeFalling>vsyncTimes[0]]]))
    # find break in vsyncs to isolate the initial, behavior block
    stimBreak = np.where(np.diff(vsyncFalling)>30)[0]
    
    # get stim running signal (not always present)
    stimRunningRising,stimRunningFalling = probeSync.get_sync_line_data(syncDataset,'stim_running')
    
    if len(stimBreak)==0:
        stimBreak = len(vsyncFalling)-1
    else:
        stimBreak = stimBreak[0]
        
    #RF mapping frame delay
    if RF_first:
        vsyncRF = vsyncTimes[(vsyncTimes>stimRunningRising[0])&
                             (vsyncTimes<=stimRunningFalling[0])]
        photodiodeRF = photodiodeAll[(photodiodeAll>vsyncRF[0])]
        photodiodeRF = photodiodeRF[:len(vsyncRF)]
    else:
        if len(stimRunningRising)>0:
            vsyncRF = vsyncTimes[(vsyncTimes>stimRunningRising[1])]
            photodiodeRF = photodiodeAll[(photodiodeAll>vsyncRF[0])]
        else:
            #only early pilot recordings
            vsyncRF = vsyncTimes[stimBreak+1:]
            photodiodeRF = photodiodeAll[np.where(photodiodeAll>vsyncRF[0])[0][0]:]
            # photodiodeRF = photodiodeAll[photodiodeAll>vsyncRF[0]:]
        photodiodeRF = photodiodeRF[:len(vsyncRF)]
    # average white & black photodiode transitions and apply to the last 2 trials?
    RFframeDelayAvg = np.zeros((len(photodiodeRF)))
    for ff in range(1,len(photodiodeRF),2):
        RFframeDelayAvg[ff-1:ff+1] = np.mean(photodiodeRF[ff-1:ff+1]-vsyncRF[ff-1:ff+1])
    
    #RF mapping sound delay
    nTrials = len(rf_df)
    trialStim = rf_df['trialStimType'].values
    stimStartFrame = rf_df['stimStartFrame']
    
    #find stimulus latencies
    nidaqDatPath = os.path.join(ephysPath,'continuous',syncData['nidaq']['dirName'],'continuous.dat')
    
    numAnalogCh = 8
    nidaqData = np.memmap(nidaqDatPath,dtype='int16',mode='r')    
    nidaqData = np.reshape(nidaqData,(int(nidaqData.size/numAnalogCh),-1)).T
    
    speakerCh = 1
    microphoneCh = 3
    microphoneData = nidaqData[microphoneCh]
    
    stimLatency = np.zeros(nTrials)
    stimStartTime = np.zeros(nTrials)
    preTime = 0.15
    postTime = 0.15
    for trial,stim in enumerate(trialStim.astype('str')):
        startFrame = stimStartFrame[trial]
        startTime = vsyncRF[startFrame] #+ syncData['nidaq']['shift']
        if 'sound' in stim:
            soundStartTime = startTime + syncData['nidaq']['shift']
            stimDur = rf_soundDur
            startSample = int((soundStartTime - preTime) * syncData['nidaq']['sampleRate'])
            endSample = int((soundStartTime + stimDur + postTime) * syncData['nidaq']['sampleRate'])
            t = np.arange(endSample-startSample) / syncData['nidaq']['sampleRate'] - preTime
            sound = rf_trialSoundArray[trial]
            tInterp = np.arange(-preTime,preTime+stimDur+postTime,1/soundSampleRate)
            mic = microphoneData[startSample:endSample]
            micInterp = np.interp(tInterp,t,mic)
            c = np.correlate(micInterp,sound,'valid')
            stimLatency[trial] = tInterp[np.argmax(c)]
            stimStartTime[trial] = tInterp[np.argmax(c)]+startTime
    
        elif 'vis' in stim:
            stimLatency[trial] = RFframeDelayAvg[trial] # approximately
            stimStartTime[trial] = RFframeDelayAvg[trial]+startTime
    
        else:
            stimStartTime[trial] = startTime
    
    rf_df['stimStartTime'] = stimStartTime
    rf_df['stimLatency'] = stimLatency
    
    if len(rf_deltaWheelPos)>0:
        RFframes = {
            'vsyncTimes':vsyncRF,
            'frameDelay':RFframeDelayAvg,
            'runningSpeed':rf_deltaWheelPos[:len(vsyncRF)]
            }
    else:
        RFframes = {
            'vsyncTimes':vsyncRF,
            'frameDelay':RFframeDelayAvg,
            }
    
    rf_frames_df=pd.DataFrame.from_dict(RFframes)
    
    return rf_df, rf_frames_df
    
# %%
def align_spike_times(ephysPath, syncData, probeNames, probeDirNames, startTime):
    
    # assign numbers to each probe letter name for making unique unit IDs       
    probe_num={
        'A':1,
        'B':2,
        'C':3,
        'D':4,
        'E':5,
        'F':6,
        }
            
    # unit data dictionary
    unitData = {
        'id':[],
        'quality':[],
        'peak_channel':[],
        'waveform_duration':[], ###note: currently uses template, not actual waveform
        # 'peakChanDepth':[],
        'probe':[],
        'amplitude':[]
        }
    spike_times = {}
    unit_templates = {}
    
    # loop through probes in this recording
    for probe,dirName in zip(probeNames,probeDirNames):
        dirPath = os.path.join(ephysPath,'continuous',dirName)
        #load kilosort output for this probe
        kilosortData = {key: np.load(os.path.join(dirPath,key+'.npy')) for key in ('spike_clusters',
                                                                                   'spike_times',
                                                                                   'templates',
                                                                                   'spike_templates',
                                                                                   'channel_positions',
                                                                                   'amplitudes')}
        # load cluster IDs
        clusterIDs = pd.read_csv(os.path.join(dirPath,'cluster_KSLabel.tsv'),sep='\t')
        # load unit metrics
        unit_metrics = pd.read_csv(glob.glob(os.path.join(dirPath,'*metrics*.csv'))[0])
        
        # create unique unit IDs
        unitIDs = np.unique(kilosortData['spike_clusters'])
        unique_unitIDs = unitIDs + probe_num[probe]*10000 + int(startTime[2:8]+startTime[9:11])*100000
        
        # concat unit metrics from multiple probes
        if probe == probeNames[0]:
            unit_metrics_all = unit_metrics
        else:
            unit_metrics_all = pd.concat([unit_metrics_all,unit_metrics],axis=0)
    
        for iu,u in enumerate(unitIDs):
            uind = np.where(kilosortData['spike_clusters']==u)[0]
            u = str(u)
            unitData['id'].append(unique_unitIDs[iu]) 
            unitData['quality'].append(clusterIDs[clusterIDs['cluster_id']==int(u)]['KSLabel'].tolist()[0])
            
            # save aligned spike times to dictionary
            spike_times[unique_unitIDs[iu]] = kilosortData['spike_times'][uind].flatten() / syncData[probe]['sampleRate'] - syncData[probe]['shift']
            
            # choose 1000 spikes with replacement, then average their templates together
            chosen_spikes = np.random.choice(uind,1000)
            chosen_templates = kilosortData['spike_templates'][chosen_spikes].flatten()
            unit_template = np.mean(kilosortData['templates'][chosen_templates],axis=0)
            
            unit_amplitude = np.median(kilosortData['amplitudes'][uind])*0.195
            
            unitData['amplitude'].append(unit_amplitude)
            
            unit_templates[unique_unitIDs[iu]]=unit_template
            
            peakChan = np.unravel_index(np.argmin(unit_template),unit_template.shape)[1]
            
            unitData['peak_channel'].append(peakChan)
            # unitData['peakChanDepth'].append(kilosortData['channel_positions'][peakChan][1])
            
            template = unit_template[:,peakChan]
            if any(np.isnan(template)):
                unitData['waveform_duration'].append(np.nan)
            else:
                peakInd = np.argmin(template)
                unitData['waveform_duration'].append(np.argmax(template[peakInd:])/(syncData[probe]['sampleRate'][0]/1000))
                
            unitData['probe'].append(probe)
                
    # create unit dataframe
    unitData_df = pd.DataFrame.from_dict(unitData)
    # concat with unit metrics
    unitData_df = pd.concat([unitData_df,unit_metrics_all.reset_index()],axis=1)
    # set unit id as the dataframe index
    unitData_df = unitData_df.set_index('id')
    
    unitData_df = unitData_df.drop(['Unnamed: 0'],axis='columns')
    
    return unitData_df, spike_times, unit_templates

# %%
def load_lick_times(syncPath):
    
    syncDataset = sync.Dataset(syncPath)
    
    lick_times = probeSync.get_sync_line_data(syncDataset,'lick_sensor')
    
    return lick_times


