from .DR_processing_script import process_ephys_sessions, infer_exp_meta


if __name__ == "__main__":
    import logging
    import argparse

    logging.basicConfig()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("session_output_dir", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("--v", "--verbose", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    mouse_id, session_date = infer_exp_meta(args.session_output_dir)

    logger.debug("Inferred exp info. mouse_id=%s session_date=%s" %
                 (mouse_id, session_date))

    logger.info("Processing session.")

    process_ephys_sessions(
        args.session_output_dir,
        mouse_id,
        session_date,
        False,
        args.output_dir,
    )

    logger.info("Processed session.")
