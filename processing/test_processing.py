import pathlib
import pytest
import hashlib

# from processing.DR_processing_script import process_ephys_sessions
from DR_processing_script import process_ephys_sessions


def generate_checksum(filepath: str) -> hashlib._Hash:
    with open(filepath, "r") as f:
        contents = f.read()

    return hashlib.md5(contents)


MAIN_PATH = pathlib.Path(
    r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\Task 2 pilot\DRpilot_644864_20230201"
)
MOUSE_ID = "644864"
SESSION_DATE = ""
EXP_NUMS = (1, 2, )


@pytest.mark.skipif(MAIN_PATH.exists(), reason="Required test inputs cannot be found.")
def test_basic(temp_dir):
    """Computes the rf mapping and trials tables and verifies that their 
    contents match a previously computed result
    """
    raise Exception("bur")
    # process_ephys_sessions(
    #     mm, mouseID, exp_nums[im], session_date, False)
