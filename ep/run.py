import multiprocessing as mp
import os
from pathlib import Path
import shutil

from eppy.runner.run_functions import multirunner
from eppy.runner.run_functions import prepare_run


def _multirunner(args):
    files, option = args
    clear = option.pop('clear', False)
    preserve = option.pop('preserve')

    multirunner((files, option))

    if clear and preserve:
        outdir = Path(option['output_directory'])
        prefix = option['output_prefix']

        for file in outdir.glob(f'{prefix}*'):
            name = file.name
            if not any(name.endswith(x) for x in preserve):
                file.unlink()


def run_idfs(jobs, processors=1):
    """Wrapper for run() to be used when running IDF5 runs in parallel.

    Parameters
    ----------
    jobs : iterable
        A list or generator made up of an IDF5 object and a kwargs dict
        (see `run_functions.run` for valid keywords).
    processors : int, optional
        Number of processors to run on (default: 1). If 0 is passed then
        the process will run on all CPUs, -1 means one less than all CPUs, etc.
    """
    if processors <= 0:
        processors = max(1, mp.cpu_count() - processors)

    shutil.rmtree("multi_runs", ignore_errors=True)
    os.mkdir("multi_runs")

    prepared_runs = (
        prepare_run(run_id, run_data) for run_id, run_data in enumerate(jobs))

    with mp.Pool(processors) as pool:
        if not hasattr(jobs, '__len__'):
            # This avoids materializing all of jobs as a list, potentially
            # use a lot of memory. Since we don't care about the returned
            # results we can use unordered, which is possibly more efficient.
            for _ in pool.imap_unordered(_multirunner, prepared_runs):
                pass  # force the whole result to be generated
        else:
            pool.map(multirunner, prepared_runs)

    shutil.rmtree("multi_runs", ignore_errors=True)
