import pathlib
import pytest
import hashlib

from .DR_processing_script import process_ephys_sessions, infer_exp_meta


def generate_checksum(filepath: str) -> bytes:
    with open(filepath, "r") as f:
        contents = f.read()

    return hashlib.md5(contents.encode("utf-8")).digest()


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

    main_path = str(MAIN_PATH)
    mouse_id, session_date = infer_exp_meta(main_path)

    print("Inferred: mouse_id=%s session_date=%s" % (mouse_id, session_date, ))

    process_ephys_sessions(
        main_path, mouse_id, session_date, False, str(output_dir))

    trials_table_path = output_dir / "trials_table.csv"
    assert trials_table_path.exists(), \
        "Trials table should exist where we expect it to."

    rf_mapping_table_path = output_dir / "rf_mapping_trials.csv"

    assert rf_mapping_table_path.exists(), \
        "RF mapping table should exist where we expect it to."

    assert generate_checksum(str(EXPECTED_TRIALS_TABLE_PATH)) == \
        generate_checksum(str(trials_table_path)), \
        "Trials table content should match expected result"

    assert generate_checksum(str(EXPECTED_RF_MAPPING_TABLE_PATH)) == \
        generate_checksum(str(rf_mapping_table_path)), \
        "RF mapping table content should match expected result"
