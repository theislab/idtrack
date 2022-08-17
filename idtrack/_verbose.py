#!/usr/bin/env python3

# Kemal Inecik
# k.inecik@gmail.com

import logging
import sys


def logger_config():
    """Todo."""
    logging.basicConfig(
        level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S", format="%(asctime)s %(levelname)s:%(name)s: %(message)s"
    )


def progress_bar(
    iteration: int, total: int, prefix: str = "Progress:", suffix: str = "", decimals: int = 2, bar_length: int = 20
):
    """This function should be called inside of loop, gives the loop's progress.

    Args:
        iteration: Current iteration.
        total: Total iteration.
        prefix: String to be placed before progress bar.
        suffix: String to be placed after progress bar.
        decimals: Number of decimals in percent complete.
        bar_length: Character length of bar.
    """
    filled_length = int(round(bar_length * iteration / float(total)))
    percents = round(100.00 * (iteration / float(total)), decimals)
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    sys.stdout.write("\r{} |{}| {}{} {}".format(prefix, bar, percents, "%", suffix)),
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write("\n")
        sys.stdout.flush()
