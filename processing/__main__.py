from DR_processing_script import process_ephys_sessions


if __name__ == "__main__":
    import logging
    import argparse

    logging.basicConfig()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("session_output_dir", type=str)
    parser.add_argument("output_dir", type=str)

    args = parser.parse_args()

    mouse_id = [x for x in args.session_output_dir.split(
        '_') if len(x) == 6 and x.isdigit()][0]
    rem_dashes = args.session_output_dir.replace('-', '')
    rem_dashes = rem_dashes.replace('\\', '_')
    session_date = [x for x in rem_dashes.split(
        '_') if len(x) == 8 and x.isdigit()]

    if len(session_date) > 0:
        session_date = session_date[0]

    logger.info("Processing session.")
    process_ephys_sessions(
        args.session_output_dir,
        mouse_id,
        session_date,
        False,
        args.output_dir,
    )
    logger.info("Processed session.")
