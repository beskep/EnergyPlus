import click
from loguru import logger

from ep.ep import save_as_eppy
from ep.gr import GRRunner
from ep.utils import set_logger

_path = click.Path(exists=True, file_okay=True, dir_okay=False)


@click.group()
@click.option('--debug', '-d', is_flag=True)
def cli(debug):
    set_logger(level=(10 if debug else 20))


@cli.command()
@click.option('--output', '-o')
@click.argument('idd')
@click.argument('idf')
def reformat(output, idd, idf):
    save_as_eppy(idd=idd, idf=idf, output=output)


@cli.command()
@click.option('--run/--no-run',
              default=True,
              show_default=True,
              help='`no-run`이면 디버그를 위해 Energy+ 시뮬레이션 실행하지 않음.')
@click.argument('case', type=_path)
def run(run, case):
    logger.info('run: {}', run)
    logger.info('case setting: "{}"', case)

    runner = GRRunner(case=case)
    runner.run(run=run)

    if run:
        runner.summarize()


@cli.command()
@click.argument('case', type=_path)
def summarize(case):
    logger.info('case setting: "{}"', case)

    runner = GRRunner(case=case)
    runner.summarize()


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    cli()
