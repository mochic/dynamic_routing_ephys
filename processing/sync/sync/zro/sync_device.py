"""
sync_device.py

Allen Institute for Brain Science

created on 14 Dec 2015

@author: derricw

ZRO device for controlling sync.

"""
from six.moves import cPickle as pickle
from shutil import copyfile
import os
import logging
import argparse
import datetime
import yaml

from zro import RemoteObject

from sync import __version__
try:
    from sync.sync import Sync
    HAS_NI = True
except NotImplementedError:
    HAS_NI = False


BASE_DIR = os.path.join(os.path.expanduser("~"), ".sync")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DEFAULT_OUTPUT = os.path.join(OUTPUT_DIR, "test")
for folder in [BASE_DIR, OUTPUT_DIR]:
    if not os.path.isdir(folder):
        os.makedirs(folder)


class FakeOutput(object):
    def __init__(self, path):
        self.path = path+".txt"
        self.file = open(path, "w")

    def start(self):
        self.file.write("START: {}\n".format(datetime.datetime.now()))
        logging.info("Dummy file created @ {}".format(self.path))

    def stop(self):
        self.file.write("STOP: {}\n".format(datetime.datetime.now()))
        self.file.close()
        logging.info("Dummy file closed @ {}".format(self.path))



class SyncDevice(RemoteObject):
    """
 
    """
    def __init__(self, rep_port, dummy=False):
        super(SyncDevice, self).__init__(rep_port=rep_port)
        self.dummy = dummy

    def init(self):
        """
        """
        self.device = 'Dev1'
        self.counter_input = 'ctr0'
        self.counter_output = 'ctr2'
        self.counter_bits = 32
        self.event_bits = 24
        self.pulse_freq = 100000.0
        self.output_path = DEFAULT_OUTPUT
        self.line_labels = "[]"
        self.delete_on_copy = False
        logging.info("Device initialized...")

    def echo(self, data):
        """
        For testing. Just echos whatever string you send.
        """
        return data

    def start(self):
        """
        Starts an experiment.
        """
        logging.info("Starting experiment...")
        if self.dummy:
            self.sync = FakeOutput(self.output_path)
        else:
            self.sync = Sync(device=self.device,
                                    bits=self.event_bits,
                                    output_path=self.output_path,
                                    freq=self.pulse_freq,
                                    buffer_size=10000,
                                    verbose=True,)

            lines = eval(self.line_labels)
            for index, line in enumerate(lines):
                self.sync.add_label(index, line)

        self.sync.start()

    def stop(self, h5_path=""):
        """
        Stops an experiment and clears the NIDAQ tasks.
        """
        logging.info("Stopping experiment...")
        try:
            self.sync.stop()
        except Exception as e:
            logging.warning(e)

        if not self.dummy:
            self.sync.clear(h5_path)
        
        self.sync = None
        del self.sync

    def load_config(self, path):
        """
        Loads a configuration from a yaml file.
        """
        logging.info("Loading configuration: %s" % path)

        with open(path, 'rb') as f:
            config = yaml.load(f)

        logging.info("Config loaded: {}".format(config))

        self.device = config['device']
        #self.counter_input = config['counter']
        #self.counter_output = config['pulse']
        #self.counter_bits = int(config['counter_bits'])
        self.event_bits = int(config.get('event_bits', 32))
        self.pulse_freq = float(config.get('freq', 100000.0))
        self.output_path = config['output_dir']
        self.line_labels = str(config['labels'])

    def save_config(self, path):
        """
        Saves a configuration to a yaml file.
        """
        logging.info("Saving configuration: %s" % path)

        config = {
            'device': self.device,
            'counter': self.counter_input,
            'pulse': self.counter_output,
            'freq': self.pulse_freq,
            'output_dir': self.output_path,
            'labels': eval(self.line_labels),
            'counter_bits': self.counter_bits,
            'event_bits': self.event_bits,
        }

        with open(path, 'wb') as f:
            yaml.dump(config, f)

    def copy_arbitrary_file(self, source, destination):
        """
        Copy an arbitrary file to a specified path.

        (source, destination)

        """
        logging.info('Copying file:\n %s -> %s' % (source, destination))
        copyfile(source, destination)
        logging.info("... Finished!")
        if self.delete_on_copy:
            os.remove(source)
            logging.info("*** Local copy removed ***")

    @property
    def platform_info(self):
        """ wse2 api requirement
        """
        info = super(SyncDevice, self).platform_info
        info['version'] = __version__
        return info

    def get_state(self):
        """ wse2 api requirement
        """
        if self.sync:
            return ("BUSY", "RECORDING")
        else:
            return ("READY", "")



def main():
    logging.basicConfig(level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Starts sync remote device.')
    parser.add_argument('--dummy', help='Use dummy data acquisition.', action="store_true")
    args = parser.parse_args()

    if not HAS_NI:
        logging.warning("Failed to find NI Drivers.  Can only run in dummy mode.")
        args.dummy = True

    sync_device = SyncDevice(rep_port=5000, dummy=args.dummy)
    sync_device.run_forever()


if __name__ == "__main__":
    main()
