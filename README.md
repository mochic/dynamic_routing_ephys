# dynamic_routing_ephys

## Inputs: 

- Behavior output (.hdf5) 
- RF mapping output (.hdf5)
- Ephys data (?)
- Datajoint outputs
- Sync data

## Outputs (we care about): 

- RF mapping table (.csv)
- Trials table (.csv) 

## Initialize

```console
poetry install
```

## Tests

poetry run pytest ./tests

*Requires access to /allen*

## Notes
- Would have liked to use `pdm` but was experiencing this issue when trying to use it:
    ```
    [TypeError]: __init__() got an unexpected keyword argument 'host'
    ```
- From Ethan Mcbride (code author):

    *It's not the cleanest or best commented code but I'm happy to answer any questions and help out if things aren't clear. 

    For making a trials table, the most important functions are: 

    load_behavior_data - loads the hdf5 file and starts making a trials table (currently doesn't save every variable from the hdf5 file to the table, we will want to save all of them eventually) 

    load_rf_mapping - loads the receptive field mapping hdf5 file - similar to above but simpler 

    sync_data_streams - aligns ephys data and NIDAQ with sync (NIDAQ alignment important for sound trial alignment at the moment. Since we're changing how sound is presented, this may become unnecessary for trials table) 

    align_trial_times - uses vsyncs and sound recordings to find stimulus start times & frame times in general, makes trials table. The method of choosing which vsyncs belong to which stimulus could be improved, currently have to tell it whether the RF mapping stimulus was first or second. Corbett has said he has a better way of doing this. 

    Currently outputs to a network folder i.e. \\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\Task 2 pilot\DRpilot_644864_20230130\processed*