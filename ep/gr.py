"""서기연 GR 사용량 분석"""
from itertools import product
from pathlib import Path
from typing import Iterable, Optional

from eppy.bunch_subclass import EpBunch
from loguru import logger
import pandas as pd
import yaml

from ep.ep import EnergyPlusCase
from ep.utils import StrPath
from ep.utils import track


def _path(path, root: Path):
    if not path:
        return None

    path = Path(path)
    if not path.exists():
        path = root.joinpath(path)

    return path.resolve()


class GRCase(EnergyPlusCase):
    pass


class GRRunner:
    UVALUE = 'uvalue'

    def __init__(self, case: StrPath) -> None:
        case = Path(case)
        root = case.parent

        with case.open('r', encoding='utf-8') as f:
            option = yaml.safe_load(f)

        self._option = option

        paths = {k: _path(v, root) for k, v in option['path'].items()}
        for k, v in paths.items():
            if k != 'output_template' and not v.exists():
                raise FileNotFoundError(f'{k} not found: {v}')
            logger.debug('{}: "{}"', k, v)

        self._paths = paths
        self._material: pd.DataFrame = pd.read_csv(
            paths['material']).set_index('year')

        self._material_names = set(self._material.columns)
        self._material_names.remove(self.UVALUE)

    def case(self):
        files = ('idd', 'idf', 'epw', 'output_template')
        case = GRCase(**{k: v for k, v in self._paths.items() if k in files})

        if self._paths['output_template'] is not None:
            # XXX 무슨 역할인지 모르겠음
            case.set_output()

        for x in self._option['remove_objs']:
            case.remove_obj(x)

        return case

    def change_year(self, year: int, case: Optional[EnergyPlusCase] = None):
        try:
            row = self._material.loc[year, :]
        except KeyError as e:
            raise ValueError(f'연도 설정 오류: {year}') from e

        if case is None:
            case = self.case()

        case.change_year(materials={x: row[x] for x in self._material_names},
                         u_value=row[self.UVALUE])

        return case

    def _run_case(self,
                  case: GRCase,
                  outdir: Path,
                  year,
                  save_idf=True,
                  run=True):
        self.change_year(year=year, case=case)
        case_name = f'ep_year{year}'

        if save_idf:
            case.idf.saveas(outdir.joinpath(f'{case_name}.idf').as_posix())

        if run:
            case.run(output_directory=outdir.as_posix(),
                     output_prefix=f'{case_name}_',
                     output_suffix=self._option['EP']['output_suffix'],
                     verbose=self._option['EP']['verbose'])

    def run(self, save_idf=True, run=True):
        case = self.case()
        years = tuple(self._option['case']['year'])
        outdir: Path = self._paths['output']

        for year in track(years):
            self._run_case(case=case,
                           outdir=outdir,
                           year=year,
                           save_idf=save_idf,
                           run=run)
