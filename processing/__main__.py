from DR_processing_script import process_ephys_sessions


def infer_exp_meta(session_output_dir: str) -> tuple[str, str]:
    """Temp solution for a potentially better solution that doesnt rely
    on the structure of a filename

    Returns
    -------
    tuple
        mouse_id
        session_date: serialized as string in format MMDDYYYY
    """
    mouse_id = [x for x in session_output_dir.split(
        '_') if len(x) == 6 and x.isdigit()][0]
    rem_dashes = session_output_dir.replace('-', '')
    rem_dashes = rem_dashes.replace('\\', '_')
    session_date = [x for x in rem_dashes.split(
        '_') if len(x) == 8 and x.isdigit()]

    if len(session_date) > 0:
        session_date = session_date[0]

    return mouse_id, session_date


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
