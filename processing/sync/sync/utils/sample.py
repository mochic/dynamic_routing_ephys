"""
sample.py

Writes sample data.  For use in testing, analysis.
"""
import numpy as np

SHORT_DATA = [0, 0, 1, 1, 0, 0, 2, 2, 0, 0]
# 4 * 2000000 * 32 bits = 32Mb
LONG_DATA = [[[0] * 1000000, [1] * 1000000] * 4]


def write_sample_data(path, data):
    data = np.array(data, dtype=np.uint32).flatten()
    with open(path, 'wb') as f:
        f.write(data.astype(np.uint32).tostring())
