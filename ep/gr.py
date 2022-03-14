"""서기연 GR 사용량 분석"""

from itertools import product
from pathlib import Path
from typing import Optional

from eppy.runner.run_functions import EnergyPlusRunError
from loguru import logger
import pandas as pd
import yaml

from .ep import EnergyPlusCase
from .utils import StrPath
from .utils import track


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
                  year: int,
                  occupancy: float,
                  lighting_level: float,
                  save_idf=True,
                  run=True):
        case_name = f'ep_year{year}_occupancy{occupancy}_lighting{lighting_level}'
        logger.info('year {} | occupancy {} people/m² | lighting {} W/m²', year,
                    occupancy, lighting_level)

        self.change_year(year=year, case=case)
        case.change_occupancy(density=occupancy)
        case.change_lighting_level(power=lighting_level)

        if save_idf:
            case.idf.saveas(outdir.joinpath(f'{case_name}.idf').as_posix())

        if run:
            case.run(output_directory=outdir.as_posix(),
                     output_prefix=case_name,
                     output_suffix=self._option['EP']['output_suffix'],
                     verbose=self._option['EP']['verbose'])

    def run(self, save_idf=True, run=True):
        case = self.case()
        outdir: Path = self._paths['output']

        year = tuple(self._option['case']['year'])
        occupancy = tuple(self._option['case']['occupancy'])
        lighting_level = tuple(self._option['case']['lighting_level'])

        it = track(product(year, occupancy, lighting_level),
                   total=(len(year) * len(occupancy) * len(lighting_level)))
        for yr, oc, ll in it:
            try:
                self._run_case(case=case,
                               outdir=outdir,
                               year=yr,
                               occupancy=oc,
                               lighting_level=ll,
                               save_idf=save_idf,
                               run=run)
            except EnergyPlusRunError as e:
                logger.exception(e)
                return
