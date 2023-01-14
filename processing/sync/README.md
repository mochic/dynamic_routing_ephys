AIBS Sync Software Package
==========================

Created 10/8/14 by Derric Williams

This program is designed to log high-speed digital IO events using National Instruments hardware and the NIDAQmx API.

It requires a board with hardware-timed digital IO lines.  One data point is taken for each rising or falling edge of the 32 Digital IO channels.

Each data point consists of two* 32-bit unsigned integers.

The first integer is the counter value that is used as the timebase.  The second 32-bit integer is the packed value of all 32 digital input lines.  For an example of how to read/parse the datasets, see below.

### Dependencies

You can pip install just the requirements with `pip install -r requirements.txt`. Dependencies for the various remote interfaces or data acquisition are not installed by default. Links are included here for reference.

### Installation

For best results, use a virtual environment.

    $ conda create -n sync python=2
    $ activate sync

    $ cd <sync directory>
    $ pip install .

If you want to acquire data or use the `sync_gui` entry point, install the acquisition requirements.

    $ pip install -r requirements_acq.txt

## Use

### Collecting Data

Import for use in your own application:

    >>> from sync import Sync

    >>> s = Sync(device='Dev1',
    ...          freq=100000,
    ...          output_path="test")

    >>> s.start()

    >>> #collect data

    >>> s.clear()

### Reading Datasets

Datasets are initially written to a binary file as unsigned 32-bit integers.  Upon close, the raw data is transfered to an HDF5 file along with all metadata.

I have made a very simple class for opening and parsing the data sets.

    >>> from sync import Dataset

    >>> dset = Dataset("test.h5")

    >>> bit0 = dset.get_bit(0)  #gets all data from bit 0

    >>> bit0_rising = dset.get_rising_edges(0)  #gets all rising edges for bit 0

    >>> stats = dset.line_stats(0)  #gets quick statistics for bit 0

    >>> stats = dset.plot_bit(0, start_time=5.0, end_time=10.0)  #plot bit 5

    >>> dset.close()

## Future Plans

1. Chunk digital event detection so that it doesn't use up all of system memory when it converts the binary data to hdf5.

## Known issues

