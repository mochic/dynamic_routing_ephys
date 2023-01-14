"""
test_sync.py

Tests the functionality of sync but with faked DAQ IO.
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

from sync.utils.sample import write_sample_data, SHORT_DATA

real_import = __import__


def mock_import_daq(mod, *args):
    if "PyDAQmx" in mod:
        return mock.MagicMock()
    elif "ni.daq" in mod:
        froms = args[2]
        if froms is not None and "DigitalInputU32" in froms:
            return mock.MagicMock()
    return real_import(mod, *args)


def test_sync(tmpdir):
    # make sure sync can create an hdf5 file out of fake binary data
    output_path = str(tmpdir) + "/test_output"
    with mock.patch(builtins.__name__+".__import__",
                    side_effect=mock_import_daq):
        from sync.sync import Sync
        sync = Sync('Dev1', 32, output_path, 10000.0)
        sync.start()
        write_sample_data(output_path, SHORT_DATA)
        sync.clear()

    # make sure it is readable
    from sync import Dataset
    ds = Dataset(output_path+".h5")
    assert ds.get_rising_edges(0) == 1
    assert ds.get_falling_edges(0) == 3
    assert ds.get_rising_edges(1) == 5
    assert ds.get_falling_edges(1) == 7
