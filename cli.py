import click
from loguru import logger

from ep.gr import GRRunner
from ep.utils import set_logger

_path = click.Path(exists=True, file_okay=True, dir_okay=False)


@click.command()
@click.option('--debug', '-d', is_flag=True)
@click.option('--run/--no-run',
              default=True,
              show_default=True,
              help='`no-run`이면 디버그를 위해 Energy+ 시뮬레이션 실행하지 않음.')
@click.argument('case', type=_path)
def main(debug, run, case):
    set_logger(level=(10 if debug else 20))

    logger.info('run: {}', run)
    logger.info('case: {}', case)

    runner = GRRunner(case=case)
    runner.run(save_idf=True, run=run)


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
