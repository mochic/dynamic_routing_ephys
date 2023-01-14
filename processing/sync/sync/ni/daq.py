"""
daq.py
"""
import sys
import logging
import os

from ctypes import c_long, c_ulong, CFUNCTYPE, POINTER
from ctypes import create_string_buffer, c_double, c_void_p, c_char_p

import PyDAQmx
from PyDAQmx import Task
from PyDAQmx.DAQmxConstants import *
from PyDAQmx.DAQmxFunctions import *
from PyDAQmx.DAQmxFunctions import function_dict, function_list
import PyDAQmx.DAQmxFunctions as DAQmxFunctions
from numpy import zeros, sin, arange, pi, array, ones
import numpy as np


##############################################################################
# NI System Object
##############################################################################

system_function_list = [name for name in function_dict.keys() if
                        "DAQmxGetSys" in name]


def encode_str_py2(string):
    return string


def encode_str_py3(string):
    return string.encode('ascii')


if sys.version_info[0] < 3:
    encode_str = encode_str_py2
else:
    encode_str = encode_str_py3


def _create_system_method(func):
    """
    Creates a System class method from a NIDAQmx function.
    """

    def _call_method(self, *args):
        return func(*args)
    return _call_method


def _create_system_buffer_method(func):
    """
    Creates a System class method from a NIDAQmx function designed to
        parse a buffer.
    """

    def _call_method(self):
        buff = encode_str(" " * self.buffer_size)
        func(buff, self.buffer_size)
        data = buff.strip().strip("\x00").split(', ')
        if data[0] == '':
            data.remove('')
        return data
    return _call_method


class System(object):
    """
    System state tracking.

    Autopopulated with the PyDAQmx methods associated with the system state.

    Added convenience methods as well for pythonicness.

    Examples:
        >>> s = System()
        >>> s.getDevNames()
        ['Dev1', 'Dev2']

    """

    def __init__(self):
        super(System, self).__init__()
        self.buffer_size = 4096

    def _get_property_u32(self, method):
        data = c_ulong()
        method(data)
        return data

    def getNIDAQVersion(self):
        major = self._get_property_u32(self.GetSysNIDAQMajorVersion).value
        minor = self._get_property_u32(self.GetSysNIDAQMinorVersion).value
        update = self._get_property_u32(self.GetSysNIDAQUpdateVersion).value
        return "{}.{}.{}".format(major, minor, update)


# Here we add functions to the System class
#   Functions with a char buffer for the first object are properties whose
#   values are written to long buffers so we given them a helper function
#   so that the user doesn't have to deal with it.
for function_name in system_function_list:
    name = function_name[5:]
    func = getattr(DAQmxFunctions, function_name)

    arg_names = function_dict[function_name]['arg_name']
    arg_types = function_dict[function_name]['arg_type']

    if len(arg_types) > 0 and (arg_types[0] is c_char_p):
        system_func = _create_system_buffer_method(func)
        name = name.replace("GetSys", "get")
    else:
        system_func = _create_system_method(func)
    system_func.__name__ = name
    system_func.__doc__ = 'S.%s(%s) -> error.' % \
        (name, ', '.join(arg_names[1:]))
    setattr(System, name, system_func)

#clean namespace a bit
del _create_system_method
del system_function_list

##############################################################################
# Device Object
##############################################################################

device_func_list = [name for name in function_dict.keys() if
                    len(function_dict[name]['arg_type']) > 0 and \
                    #(function_dict[name]['arg_type'][0] in c_char_p) and \
                    'device' in function_dict[name]['arg_name'][0]]


def _create_device_method(func):
    """
    Creates a System class method from a NIDAQmx function.
    """

    def _call_method(self, *args):
        return func(self.device_name, *args)
    return _call_method


def _create_device_buffer_method(func):
    """
    Creates a Device class method from a NIDAQmx function designed to
        parse a buffer.
    """

    def _call_method(self):
        buff = encode_str(" " * self.buffer_size)
        func(self.device_name, buff, self.buffer_size)
        data = buff.strip().strip("\x00").split(', ')
        if data[0] == '':
            data.remove('')
        return data
    return _call_method


class Device(object):
    """
    Device object.

    Autopopulated with functions that use "deviceName" as their first argument.

    Some methods (those that start with a lower-case letter), have been
        replaced with a method that automatically builds and parses the buffer


    Args:
        device_name (str): The device name Ex: "Dev1"

    Example:
        >>> d = Device('Dev1')
        >>> d.getDOPorts()
        ['Dev1/port0', 'Dev1/port1']

    """

    def __init__(self, device_name):
        super(Device, self).__init__()
        self.device_name = device_name
        self.buffer_size = 4096

    def _get_property_buffer(self, method):
        buff = encode_str(" " * self.buffer_size)
        method(buff, self.buffer_size)
        return buff.strip().strip("\x00").split(', ')

    def getAIChannels(self):
        return self.getAIPhysicalChans()

    def getAOChannels(self):
        return self.getAOPhysicalChans()

    def getCOChannels(self):
        return self.getCOPhysicalChans()

    def getCIChannels(self):
        return self.getCIPhysicalChans()

    def reset(self):
        return self.ResetDevice()


for function_name in device_func_list:
    name = function_name[5:]
    func = getattr(DAQmxFunctions, function_name)
    arg_names = function_dict[function_name]['arg_name']
    arg_types = function_dict[function_name]['arg_type']

    if len(arg_types) == 3 and (arg_types[1] is c_char_p) and \
            (arg_types[2] is c_ulong):
        devfunc = _create_device_buffer_method(func)
        name = name.replace("GetDev", "get")
        name = name.replace("Get", "get")
    else:
        devfunc = _create_device_method(func)
    devfunc.__name__ = name
    devfunc.__doc__ = 'D.%s(%s) -> error.' % \
        (name, ', '.join(arg_names[1:]))
    setattr(Device, name, devfunc)

del _create_device_method
del device_func_list

##############################################################################
# Task Objects
##############################################################################


class BaseTask(Task):
    """
    Base class for NIDAQmx tasks.

    Base tasks aren't pre-configured for anything.  They have some convenience
        methods for clock and trigger configuration, but haven't set up any
        channels for IO yet.

    They can still use all of the methods of the PyDAQmx Task object.

    Example:
        >>> from PyDAQmx.DAQmxConstants import *
        >>> import numpy as np
        >>> bt = BaseTask()
        >>> bt.CreateDOChan('Dev1/port0/line0:4',
                           '',
                           DAQmx_Val_ChanForAllLines)
        >>> bt.start()
        >>> buf = np.array([0,1,0,1], dtype=np.uint8)
        >>> bt.WriteDigitalLines(1, 0, 10.0, DAQmx_Val_GroupByChannel, buf,
                                None, None)
        >>> bt.stop()
        >>> bt.clear()

    """

    def __init__(self):
        Task.__init__(self)  # old style class
        self.__registered = False  # data callback not registered

    def start(self):
        """
        Starts the task.
        """
        self.StartTask()

    def stop(self):
        """
        Stops the task.  It can be restarted.
        """
        self.StopTask()

    def clear(self):
        """
        Clears the task.  It cannot be restarted.
        """
        try:
            self.stop()
        except Exception as e:
            ##TODO: catch specific type
            print(e)

        self.ClearTask()

    def cfg_sample_clock(self,
                         rate,
                         source="",
                         edge='rising',
                         mode='continuous',
                         buffer_size=1000,
                         ):
        """
        Configures the sample clock.

        Args:
            rate (float): Sample rate in Hz
            source (Optional[str]): name of source terminal
            edge (Optional[str]): rising or falling edge for example "r"
            mode (Optional[str]): sample mode for example "continuous"
            buffer_size (Optional[int]): write buffer size

        Examples:
            >>> mytask.cfg_sample_clock("/Dev1/ai/SampleClock", 'f', 'c', 1000)

        """
        edge = get_edge_val(edge)
        mode = get_mode_val(mode)

        status = self.CfgSampClkTiming(source, rate, edge, mode, buffer_size)
        self.buffer_size = buffer_size
        self.clock_speed = rate
        logging.debug("Sample clock configured to ({}, {}, {}, {}, {})".format(rate,
                                                                               source, edge, mode, buffer_size))
        return status

    def cfg_dig_start_trigger(self,
                              source,
                              edge='rising',
                              ):
        """
        Configures the start trigger.

        Args:
            source (str): Start trigger source.
            edge (str): rising or falling edge

        Examples:
            >>> mytask.cfg_digital_start_trigger("/Dev1/ai/StartTrigger",'r')

        """
        edge = get_edge_val(edge)
        self.CfgDigEdgeStartTrig(source, edge)
        logging.debug(
            "Start trigger configured to ({}, {})".format(source, edge))

    def set_timebase_divisor(self, divisor=1):
        """
        Supposed to set the divisor for the clock's timebase.

        Doesn't seem to work...

        #TODO: Call NI and ask them why this doesn't work.

        """
        if divisor == 1:
            self.ResetSampClkTimebaseDiv()
        elif divisor > 1:
            self.SetSampClkTimebaseDiv(divisor)
            #print(divisor)
        else:
            raise ValueError("Divisor must be between 1 and 2^32")

    def get_clock_terminal(self):
        """
        Returns the terminal for the sample clock.

        Example output: "/Dev1/ai/SampleClock"
        """
        buffer_size = 1024
        lines = " " * buffer_size
        self.GetSampClkTerm(lines, buffer_size)
        return lines.strip().strip('\x00').split(', ')[0]

    def get_start_trigger_term(self):
        """
        Returns the terminal for start trigger.
        """
        buffer_size = 1024
        lines = " " * buffer_size
        self.GetStartTrigTerm(lines, buffer_size)
        return lines.strip().strip('\x00').split(', ')[0]

    def register_sample_callback(self,
                                 buffer_size,
                                 direction='input',
                                 synchronous=False):
        """
        Register a sample callback for a buffer of N samples.
        """
        direction = get_direction_val(direction)
        synchronous = get_synchronous_val(synchronous)
        self.AutoRegisterEveryNSamplesEvent(direction,
                                            buffer_size,
                                            synchronous)
        self.__registered = True
        logging.debug(
            "Task sample callback registered for {} samples.".format(buffer_size))

    def unregister_sample_callback(self,
                                   direction='input',
                                   synchronous=False):
        """
        Unregister a sample callback.
        """
        direction = get_direction_val(direction)
        synchronous = get_synchronous_val(synchronous)
        if self.__registered:
            self.RegisterEveryNSamplesEvent(direction,
                                            self.buffer_size,
                                            synchronous,
                                            DAQmxEveryNSamplesEventCallbackPtr(
                                                0),
                                            None)
            self.__registered = False
            logging.debug("Task sample callback unregistered.")
        else:
            logging.debug("Task already unregistered.")

##############################################################################
# DigitalOutputU32
##############################################################################

class DigitalInputU32(BaseTask):
    '''
    Like the regular digital input but reads buffers sampled as a specified
        rate.

    Parameters
    ----------

    device : str
        NIDAQ device id (ex:'Dev1')
    lines : int or str
        Lines to reserve and read data from:  32, "0:8"
    timeout : float
        Seconds to wait for samples
    clock_speed : float
        Sample clock speed
    buffer_size : int
        Length of buffer to write to disk
    binary : str
        Binary file to write to

    Returns
    -------

    DigitalInputU32 : Task
        Task object

    Examples
    --------

    >>> task = DigitalInputU32('Dev1', 32) # all 32 lines
    >>> task.start()
    >>> time.sleep(10)  #collect some data
    >>> task.clear()

    '''

    def __init__(self,
                 device='Dev1',
                 lines=32,
                 timeout=10.0,
                 clock_speed=10000.0,
                 buffer_size=1000,
                 binary=None,
                 ):

        BaseTask.__init__(self)

        self.timeout = timeout
        self.lines = lines
        self.device = device
        self.clock_speed = clock_speed
        self.buffer_size = buffer_size
        self.binary = binary

        #set up task properties
        if isinstance(lines, int):
            self.devstr = "%s/line0:%i" % (self.device, lines - 1)
        elif isinstance(lines, str):
            self.devstr = "%s/line%s" % (self.device, lines)

        #create channel
        self.CreateDIChan(self.devstr, "", DAQmx_Val_ChanForAllLines)

        #configure sampleclock
        self.cfg_sample_clock(rate=self.clock_speed,
                              edge='rising',
                              mode='continuous',
                              buffer_size=self.buffer_size)

        if self.binary is not None:
            self.outFile = open(self.binary, 'wb')
            self.samples_written = 0
            self.max_samples = 100000 * 60 * 60 * 6

    def cfg_sample_clock(self,
                         rate=10000.0,
                         source="",
                         edge='rising',
                         mode='continuous',
                         buffer_size=1000,
                         callback=True):
        """
        Custom version of the clock config function.  Needs to re-register
            the NSamples callback.
        """
        # then set up the sample clock
        BaseTask.cfg_sample_clock(self,
                                  rate=rate,
                                  source=source,
                                  edge=edge,
                                  mode=mode,
                                  buffer_size=buffer_size)

        self.data = zeros((buffer_size), dtype=np.uint32)  # data buffer

    def EveryNCallback(self):
        """
        Executed every N samples, where N is the buffer_size.  Reads the
            current buffer off of the DAQ.  Writes the samples to disk if
            a binary output file was specified.

        # This is not automatically registered. to use, run
            self.register_sample_callback(...)
        """
        self.read_buffer()
        if self.binary:
            self.write_buffer()

    def read_buffer(self):
        """ Reads the current data buffer from hardware and returns number
            of samples read.

        :return:
        """
        read = int32()
        self.ReadDigitalU32(self.buffer_size, self.timeout, DAQmx_Val_Auto,
                            self.data, self.buffer_size, byref(read), None)
        return read

    def write_buffer(self):
        """ Writes the current data buffer to a binary file.
        :return:
        """
        self.outFile.write(self.data.astype(np.uint32).tostring())
        self.samples_written += self.buffer_size
        if self.samples_written > self.max_samples:
            self.stop()
            self.clear()
            raise RuntimeError("Maximum sample count reached.")

    def clear(self):
        """
        Clears the task.  Also flushes and closes the binary file if it
            exists.
        """
        BaseTask.clear(self)
        if self.binary:
            self.outFile.flush()
            self.outFile.close()

    def DoneCallback(self, status):
        """
        Done callback.  Unregistered at this point.  Might just eliminate it.
        """
        return 0  # The function should return an integer

###########################################################################################
# AnalogInput
###########################################################################################

class AnalogInput(BaseTask):
    '''
    Gets analog input from NIDAQ device.
        Tested using several buffer sizes and channels on a NI USB-6210.

    Parameters
    ----------

    device : 'Dev1'
        NIDAQ device id
    channels : [0]
        List of channels to read
    buffer_size : 500
        Integer size of buffer to read
    clock_speed : 10000.0
        Float sample clock speed
    terminal_config : "RSE"
        String for terminal type: "RSE","Diff"
    voltage_range : [-10.0,10.0]
        Float bounds for voltages
    timout : 10.0
        Float timeout for read
    tdms : None
        tdms file to write to.
    binary : None
        binary file to write to
    dtype : np.float64
        output data type

    Returns
    -------

    AnalogInput : Task
        Task object

    Examples
    --------

    >>> ai = AnalogInput('Dev1',channels=[0],buffer_size=500)
    >>> ai.start()
    >>> for x in range(10):
    ...     time.sleep(1) #collects some data
    ...     print ai.data #prints the current buffer
    >>> ai.clear()

    '''

    def __init__(self,
                 device='Dev1',
                 channels=[0],
                 buffer_size=500,
                 clock_speed=10000.0,
                 terminal_config="RSE",
                 voltage_range=[-10.0, 10.0],
                 timeout=10.0,
                 binary=None,
                 dtype=np.float64,
                 custom_callback=None):

        BaseTask.__init__(self)

        #set up task properties
        self.buffer_size = buffer_size
        self.clock_speed = clock_speed
        if isinstance(channels, int):
            channels = [channels]
        self.channels = channels
        self.data = zeros((self.buffer_size,
                           len(self.channels)), dtype=np.float64)  # data buffer
        self.dataArray = []
        self.binary = binary
        self.terminal_config = get_input_terminal_config(terminal_config)
        self.voltage_range = voltage_range
        self.timeout = timeout
        self.dtype = dtype
        if custom_callback:
            self.callback = custom_callback
        else:
            self.callback = self.default_callback
        self.buffercount = 0

        #create dev str for various channels
        self.devstr = ""
        if type(channels) is int:
            channels = [channels]
        for channel in channels:
            self.devstr += str(device) + "/ai" + str(channel) + ","
        self.devstr = self.devstr[:-1]

        self.CreateAIVoltageChan(self.devstr, "", self.terminal_config,
                                 self.voltage_range[0], self.voltage_range[1],
                                 DAQmx_Val_Volts, None)

        self.cfg_sample_clock(rate=self.clock_speed,
                              edge='rising',
                              mode='continuous',
                              buffer_size=self.buffer_size)

        if self.binary is not None:
            self.outFile = open(self.binary, 'wb')

    def cfg_sample_clock(self,
                         rate=10000.0,
                         source="",
                         edge='rising',
                         mode='continuous',
                         buffer_size=1000):
        """
        Custom version of the clock config function.  Needs to re-register
            the NSamples callback.
        """

        # first unregister the old buffer callback if it is registered
        self.unregister_sample_callback()

        # then set up the sample clock
        BaseTask.cfg_sample_clock(self,
                                  rate=rate,
                                  source=source,
                                  edge=edge,
                                  mode=mode,
                                  buffer_size=buffer_size)

        # set up a new data buffer
        self.data = zeros((buffer_size, len(self.channels)),
                          dtype=np.float64)

        # then register the buffer callback
        self.register_sample_callback(buffer_size)

    def EveryNCallback(self):
        """
        Callback for buffer read.  Occurs automatically when `self.buffer_size`
            samples are read.
        """
        try:
            read = int32()
            # read into the data buffer
            self.ReadAnalogF64(self.buffer_size, self.timeout, DAQmx_Val_Auto,
                               self.data, (self.buffer_size *
                                           len(self.channels)), byref(read),
                               None)
            if self.binary:
                self.outFile.write(self.data.astype(self.dtype).tostring())
            self.callback(self.data)
            self.buffercount += 1
        except Exception as e:
            print("Failed to read buffer #%i -> %s" % (self.buffercount, e))

    def read(self, samples=1):
        """
        Syncrhonous read.
        """
        read = int32()
        output_size = len(self.channels) * samples
        output_array = np.zeros(
            (len(self.channels), samples), dtype=np.float64)
        self.ReadAnalogF64(samples, self.timeout, DAQmx_Val_GroupByScanNumber,
                           output_array, output_size, byref(read),
                           None)
        return output_array

    def clear(self):
        BaseTask.clear(self)
        if self.binary:
            self.outFile.flush()
            self.outFile.close()

    def default_callback(self, data):
        return


###################################################################################
# CounterOutputU32
###################################################################################

class CounterInputU32(BaseTask):
    """
    Generic edge counter for single U32 counter.

    Parameters
    ----------
    device : str
        NI DAQ ID.  Ex: "Dev1"
    counter : str
        Counter terminal.  Ex: 'ctr0'
    edge : str
        Edge to count.  Either "rising" or "falling"
    direction : str
        Counter direction.  'up' or 'down'
    initial_count : int
        Initial counter value.
    timeout: float
        Read timeout.

    """

    def __init__(self,
                 device='Dev1',
                 counter='ctr0',
                 edge='rising',
                 direction='up',
                 initial_count=0,
                 timeout=10.0,
                 ):

        BaseTask.__init__(self)
        self.device = device
        self.counter = counter
        self.edge = edge
        self.direction = direction
        self.initial_count = initial_count
        self.buffer_size = None
        self.timeout = timeout

        self.devstr = "%s/%s" % (device, counter)

        if direction.lower() == 'up':
            dir_val = DAQmx_Val_CountUp
        elif direction.lower() == 'down':
            dir_val = DAQmx_Val_CountDown
        else:
            raise KeyError("Invalid direction.  Try 'up' or 'down'.")

        self.CreateCICountEdgesChan(self.devstr, "", get_edge_val(self.edge),
                                    initial_count, dir_val)

        self.data = c_ulong()

    def read(self):
        """
        A simple scalar read of the current counter value.
        """
        self.ReadCounterScalarU32(self.timeout, self.data, None)
        return self.data

    def setup_file_output(self,
                          path=None,
                          file_type="bin",
                          buffer_size=1000,
                          ):
        """
        Sets up data output writing.  This alone is insufficient.  You must Also
            configure the sample clock.
        """
        if not path:
            self.buffer_count = None
            self.unregister_sample_callback()
            return True
        else:
            self.buffer_size = buffer_size
            self.buffer_count = 0
            self.register_sample_callback(self.buffer_size)

        self.data = np.zeros(self.buffer_size, dtype=np.uint32)

        if file_type == 'bin':
            self.output_file = open(path, 'wb')
        else:
            raise NotImplementedError(
                "file types other than binary are unimplemented.")

    def clear(self):
        super(CounterInputU32, self).clear()
        if self.output_file:
            self.output_file.close()

    def EveryNCallback(self):
        """
        Callback for buffer read.  Occurs automatically when `self.buffer_size`
            samples are read if buffered reading is enabled.
        """
        try:
            read = int32()

            # read into the data buffer
            self.ReadCounterU32(self.buffer_size, self.timeout, self.data,
                                self.buffer_size, byref(read), None)

            #self.ReadDigitalU32(self.buffer_size, self.timeout, DAQmx_Val_Auto,
            #        self.data, self.buffer_size, byref(read), None)

            self.output_file.write(self.data.tostring())
            self.buffer_count += 1
        except Exception as e:
            print("Failed to read buffer #%i -> %s" % (self.buffer_count, e))

    def getCountEdgesTerminal(self):
        """
        Returns the terminal for edge counting input (str)

        Example output: "/Dev1/PFI8"
        """
        buffer_size = 1024
        lines = " " * buffer_size
        self.GetCICountEdgesTerm(self.devstr, lines, buffer_size)
        return lines.strip().strip('\x00').split(', ')

    def setCountEdgesTerminal(self, terminal):
        """
        Sets the edge counting input terminal.

        Example input: "Ctr0InternalOutput"
        """
        self.SetCICountEdgesTerm(self.devstr, terminal)

###################################################################################
# Helper functions for constants
###################################################################################

def get_edge_val(edge):
    """
    Gets the correct edge constant for a given input.
    """
    if edge in [DAQmx_Val_Rising, DAQmx_Val_Falling]:
        pass
    elif isinstance(edge, str):
        if edge.lower() in ["falling", 'f']:
            edge = DAQmx_Val_Falling
        elif edge.lower() in ["rising", 'r']:
            edge = DAQmx_Val_Rising
        else:
            raise ValueError(
                "Only 'rising'('r') or 'falling'('f') is accepted.")
    else:
        raise ValueError(
            "Edge must be str ('falling') or int (DAQmx_Val_Falling)")
    return edge


def get_mode_val(mode):
    """
    Gets the correct mode constant for a given input.
    """
    if mode in [DAQmx_Val_FiniteSamps,
                DAQmx_Val_ContSamps,
                DAQmx_Val_HWTimedSinglePoint]:
        pass
    elif isinstance(mode, str):
        if mode.lower() in ["finite", 'f']:
            mode = DAQmx_Val_FiniteSamps
        elif mode.lower() in ["continuous", 'c']:
            mode = DAQmx_Val_ContSamps
        elif mode.lower() in ['hwtsp', 'h']:
            mode = DAQmx_Val_HWTimedSinglePoint
        else:
            raise ValueError(
                "Only 'finite'('f'), 'continuous'('c'), or 'hwtsp'('h') is accepted.")
    else:
        raise ValueError(
            "Mode must be str ('finite') or int (DAQmx_Val_FiniteSamps)")
    return mode


def get_direction_val(direction):
    """
    Gets the correct direction type for a given input.
    """
    if direction in [DAQmx_Val_Acquired_Into_Buffer,
                     DAQmx_Val_Transferred_From_Buffer, ]:
        pass
    elif isinstance(direction, str):
        if direction.lower() in ['in', 'input', 'acquired', 'acq', 'i']:
            direction = DAQmx_Val_Acquired_Into_Buffer
        elif direction.lower() in ['out', 'output', 'written', 'o']:
            direction = DAQmx_Val_Transferred_From_Buffer
        else:
            raise ValueError("Only 'input'('i') or 'output'('o') is accepted.")
    else:
        raise ValueError(
            "Direction must be str ('input') or int (DAQmx_Val_Transferred_From_Buffer).")
    return direction


def get_synchronous_val(synchronous):
    """
    Gets the correct synchronous type for a given input.
    """
    if synchronous in [0, DAQmx_Val_SynchronousEventCallbacks]:
        pass
    elif synchronous is True:
        synchronous = DAQmx_Val_SynchronousEventCallbacks
    elif synchronous is False:
        synchronous = 0
    else:
        raise ValueError(
            "Synchronous must be bool or int (DAQmx_Val_SynchronousEventCallbacks)")
    return synchronous


def get_input_terminal_config(config):
    """
    Gets the correct config value for a given input.
    """
    if config in [DAQmx_Val_Cfg_Default,
                  DAQmx_Val_RSE,
                  DAQmx_Val_NRSE,
                  DAQmx_Val_Diff,
                  DAQmx_Val_PseudoDiff, ]:
        pass
    elif isinstance(config, str):
        config = config.lower()
        if config in ['default', ]:
            config = DAQmx_Val_Cfg_Default
        elif config in ['rse', 'r']:
            config = DAQmx_Val_RSE
        elif config in ['nrse', 'n']:
            config = DAQmx_Val_NRSE
        elif config in ['diff', 'd']:
            config = DAQmx_Val_Diff
        elif config in ['pseudodiff', 'pseudo', 'p']:
            config = DAQmx_Val_PseudoDiff
        else:
            raise ValueError(
                "Invalid terminal config type. Try 'rse' or 'diff'.")
    else:
        raise ValueError(
            "Terminal config type must be str ('rse') or int (DAQmx_Val_Diff).")
    return config
