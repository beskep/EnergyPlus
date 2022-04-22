from io import StringIO
from logging import LogRecord
from os import PathLike
import re
from typing import Optional, Union

from loguru import logger
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import track as _track
from rich.theme import Theme

StrPath = Union[str, PathLike]

console = Console(theme=Theme({'logging.level.success': 'blue'}))
_BLANK_NO = 21


class _Handler(RichHandler):
    _levels = {5: 'TRACE', 25: 'SUCCESS', _BLANK_NO: ''}

    def emit(self, record: LogRecord) -> None:
        if record.levelno in self._levels:
            record.levelname = self._levels[record.levelno]

        return super().emit(record)


_handler = _Handler(console=console, log_time_format='[%X]')


def set_logger(level: Union[int, str] = 20):
    if isinstance(level, str):
        levels = {
            'TRACE': 5,
            'DEBUG': 10,
            'INFO': 20,
            'SUCCESS': 25,
            'WARNING': 30,
            'ERROR': 40,
            'CRITICAL': 50
        }
        try:
            level = levels[level.upper()]
        except KeyError as e:
            raise KeyError(f'`{level}` not in {set(levels.keys())}') from e

    if getattr(logger, 'lvl', -1) != level:
        logger.remove()

        logger.add(_handler,
                   level=level,
                   format='{message}',
                   backtrace=False,
                   enqueue=True)
        logger.add('ep.log',
                   level='DEBUG',
                   rotation='1 week',
                   retention='1 month',
                   encoding='UTF-8-SIG',
                   enqueue=True)

        setattr(logger, 'lvl', level)

    try:
        logger.level('BLANK')
    except ValueError:
        # 빈 칸 표시하는 'BLANK' level 새로 등록
        logger.level(name='BLANK', no=_BLANK_NO)


def track(sequence,
          description='Working...',
          total: Optional[float] = None,
          transient=False,
          **kwargs):
    """Track progress on console by iterating over a sequence."""
    return _track(sequence=sequence,
                  description=description,
                  total=total,
                  console=console,
                  transient=transient,
                  **kwargs)


def read_str_df(txt, **kwargs) -> pd.DataFrame:
    return pd.read_csv(StringIO(txt), **kwargs)


def read_table_csv(path, table: str, melt=True) -> pd.DataFrame:
    blank = re.compile(r'^(\s|,)*$')
    lines = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith(table):
                break

        for line in f:
            if blank.match(line):
                # 표 제목, (REPORT, FOR 등)과 숫자를
                # 나누는 빈 칸이 나올 때까지 넘김
                break

        for line in f:
            if blank.match(line):
                break
            lines.append(line.lstrip(','))

    if not lines:
        raise ValueError(f'table name not found: {table}')

    df = read_str_df(''.join(lines)).reset_index()

    if melt and len(df.columns) > 1:
        df = pd.melt(df, id_vars='index')
        # df = df.loc[df['value'] != 0, :]

    return df
