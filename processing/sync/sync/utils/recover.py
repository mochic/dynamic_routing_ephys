"""
recover.py

@author: derricw

script for recovering binary data files that were never converted to hdf5.

We mock out all the DAQ import and just use the hdf5-saving functionality of the Sync class.
In the future, this functionality should probably just be moved out of the class itself
so that no mocking is necessary.

"""
try:
    # python3
    import unittest.mock as mock
except ImportError:
    # python2
    import mock
try:
    # python2
    import __builtin__ as builtins
except ImportError:
    # python3
    import builtins
import pytest

real_import = __import__


def mock_import_daq(mod, *args, **kwargs):
    if "PyDAQmx" in mod:
        return mock.MagicMock()
    elif "ni.daq" in mod:
        froms = args[2]
        if froms is not None and "DigitalInputU32" in froms:
            return mock.MagicMock()
    return real_import(mod, *args, **kwargs)


def recover_binary_data(data_file, line_labels=[], sample_freq=100000.0):
    with mock.patch(builtins.__name__+".__import__",
                        side_effect=mock_import_daq):
        from sync.sync import Sync
        s = Sync("Dev1",
                 32,
                 output_path=bin_data,
                 freq=100000.0,
                 save_raw=True)
        
        for bit, line in enumerate(line_labels):
            s.add_label(bit, line)

        s._save_hdf5()

###
bin_data = r"\\allen\programs\braintv\workgroups\nc-ophys\Doug\Julie\Data\M348126\2018.03.13_M348126_DoC_Day4\additional_data\031318_M348126_DoC_day4180313155313"

line_labels = ['behavior_vsync',
               'behavior_sweep',
               'fluorescence_camera',
               'behavior_camera',
               'face_camera',
               'eye_camera',
               'capacitive_licks',]
###

recover_binary_data(bin_data, line_labels)