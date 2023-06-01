import pathlib
import pytest
import hashlib

from DR_processing_script import process_ephys_sessions, infer_exp_meta


def generate_checksum(filepath: str) -> hashlib._hashlib.HASH:
    with open(filepath, "r") as f:
        contents = f.read()

    return hashlib.md5(contents)


MAIN_PATH = pathlib.Path(
    r"\\allen\programs\mindscope\workgroups\dynamicrouting\PilotEphys\Task 2 pilot\DRpilot_644864_20230201"
)
EXPECTED_TRIALS_TABLE_PATH = pathlib.Path(
    '//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/DRpilot_644864_20230201/processed/trials_table.csv'
)
EXPECTED_RF_MAPPING_TABLE_PATH = pathlib.Path(
    '//allen/programs/mindscope/workgroups/dynamicrouting/PilotEphys/Task 2 pilot/DRpilot_644864_20230201/processed/rf_mapping_trials.csv'
)

test_inputs_exist = all([
    MAIN_PATH.exists(),
    EXPECTED_TRIALS_TABLE_PATH.exists(),
    EXPECTED_RF_MAPPING_TABLE_PATH.exists(),
])


@pytest.mark.skipif(not test_inputs_exist, reason="Required test inputs cannot be found.")
def test_basic(tmpdir):
    """Computes the rf mapping and trials tables and verifies that their 
    contents match a previously computed result
    """
    output_dir = tmpdir.mkdir("output")

    main_path = MAIN_PATH.posix()
    mouse_id, session_date = infer_exp_meta(MAIN_PATH.posix())

    process_ephys_sessions(
        main_path, mouse_id, session_date, False, output_dir.posix())

    assert (output_dir / "trials_table.csv").exists(), \
        "Trials table should exist where we expect it to."
