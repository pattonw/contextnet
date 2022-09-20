from .v1 import v1
from .v2 import v2

import logging
import click

import sys



logging.basicConfig(level=logging.INFO)


@click.group()
def main(args=None):
    """Console script for contextnet."""
    return None

main.add_command(v1)
main.add_command(v2)

if __name__ == "__main__":
    sys.exit(main())
