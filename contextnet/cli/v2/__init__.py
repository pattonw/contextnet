import click

from .view_snapshots import view_snapshots
from .visualize_pipeline import visualize_pipeline
from .train import train


@click.group()
def v2():
    pass

v2.add_command(view_snapshots)
v2.add_command(visualize_pipeline)
v2.add_command(train)
