import click
from loguru import logger

from ep import gr as _gr
from ep import gr2 as _gr2
from ep import utils
from ep.ep import save_as_eppy

_path = click.Path(exists=True, file_okay=True, dir_okay=False)


@click.group()
@click.option('--debug', '-d', is_flag=True)
def cli(debug):
    utils.set_logger(level=(10 if debug else 20))


@cli.command()
@click.option('--output', '-o')
@click.argument('idd')
@click.argument('idf')
def reformat(output, idd, idf):
    save_as_eppy(idd=idd, idf=idf, output=output)


@cli.command()
@click.option('--clear/--no-clear',
              'flag_clear',
              default=True,
              show_default=True)
@click.argument('path', default='output')
def clear(flag_clear, path):
    logger.debug('clear: {} | path: {}', flag_clear, path)
    utils.clear_output(path=path, clear=flag_clear)


@cli.command()
@click.option('--run/--no-run',
              default=True,
              show_default=True,
              help='`no-run`이면 디버그를 위해 Energy+ 시뮬레이션 실행하지 않음.')
@click.option('--summarize', '-s', is_flag=True, help='시뮬레이션 결과 요약만 실행.')
@click.argument('case', type=_path)
def gr(run, summarize, case):
    logger.info('case setting: "{}" | run: {} | summarize: {}', case, run,
                summarize)

    runner = _gr.GRRunner(case=case)

    if run and not summarize:
        runner.run(run=run)

    if run or summarize:
        runner.summarize()


@cli.command()
@click.option('--run/--no-run',
              default=True,
              show_default=True,
              help='`no-run`이면 디버그를 위해 Energy+ 시뮬레이션 실행하지 않음.')
@click.option('--summarize', '-s', is_flag=True, help='시뮬레이션 결과 요약만 실행.')
@click.option('--size', default=1, show_default=True)
@click.argument('case', type=_path)
def gr2(run, summarize, size, case):
    logger.info('case setting: "{}" | run: {} | summarize: {}', case, run,
                summarize)

    runner = _gr2.GRRunner(case=case)

    if run and not summarize:
        runner.run(size=size, run=run)

    if run or summarize:
        runner.summarize()


if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    cli()
