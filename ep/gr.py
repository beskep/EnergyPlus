"""서기연 GR 사용량 분석"""

from itertools import product
from pathlib import Path

from eppy.runner.run_functions import runIDFs
from loguru import logger
import pandas as pd
import yaml

from .ep import EnergyPlusCase
from .utils import read_table_csv
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


def _idf_paths(idf, root: Path):
    idf = _path(idf, root=root)

    if not idf:
        raise FileNotFoundError(idf)
    if idf.is_file():
        idfs: tuple = (idf,)
    elif idf.is_dir():
        idfs = tuple(idf.glob('*.idf'))
    else:
        raise OSError(f'IDF path error: {idf}')

    return idfs


def _other_paths(option: dict, root: Path):
    paths = {k: _path(v, root) for k, v in option['path'].items()}
    for k, v in paths.items():
        if k != 'output_template' and not v.exists():
            raise FileNotFoundError(f'{k} not found: {v}')
        logger.debug('{}: "{}"', k, v)

    return paths


class GRRunner:
    UVALUE = 'uvalue'

    def __init__(self, case: StrPath) -> None:
        case = Path(case)
        root = case.parent

        with case.open('r', encoding='utf-8') as f:
            option = yaml.safe_load(f)

        self._option = option
        self._idfs = _idf_paths(idf=self._option['idf'], root=root)
        self._paths = _other_paths(option=option, root=root)

        # material
        self._material: pd.DataFrame = pd.read_csv(
            self._paths['material']).set_index('year')

        self._material_names = set(self._material.columns)
        self._material_names.remove(self.UVALUE)

    def case(self, idf):
        files = ('idd', 'epw', 'output_template')
        case = GRCase(idf=idf,
                      **{k: v for k, v in self._paths.items() if k in files})

        if self._paths['output_template'] is not None:
            # XXX 무슨 역할인지 모르겠음
            case.set_output()

        for x in self._option['remove_objs']:
            case.remove_obj(x)

        return case

    def change_year(self, case: EnergyPlusCase, year: int):
        try:
            row = self._material.loc[year, :]
        except KeyError as e:
            raise ValueError(f'연도 설정 오류: {year}') from e

        case.change_year(materials={x: row[x] for x in self._material_names},
                         u_value=row[self.UVALUE])

        return case

    def _case_iterator(self):
        year = tuple(self._option['case']['year'])
        occupancy = tuple(self._option['case']['occupancy'])
        lighting = tuple(self._option['case']['lighting_level'])

        total = (len(self._idfs) * len(year) * len(occupancy) * len(lighting))
        it = track(product(self._idfs, year, occupancy, lighting), total=total)

        return it, total

    def _idf_iterator(self):
        last_idf, case = None, None
        outdir: Path = self._paths['output']

        it, total = self._case_iterator()
        logger.info('total {} cases', total)

        for idf, yr, oc, ll in it:
            if last_idf != idf:
                case = self.case(idf=idf)

            name = f'{Path(idf).stem}_year{yr}_occupancy{oc}_lighting{ll}'

            self.change_year(case=case, year=yr)
            case.change_occupancy(oc)
            case.change_lighting_level(ll)

            case.idf.saveas(outdir.joinpath(f'{name}.idf'))
            version = '-'.join(case.idf_version()[:3])

            option = dict(output_directory=outdir.as_posix(),
                          output_prefix=name,
                          output_suffix='D',
                          verbose=self._option['EP']['verbose'],
                          readvars=self._option['EP']['readvars'],
                          ep_version=version)

            yield case.idf, option

    def run(self, run=True):
        if run:
            processors = int(self._option['processors'])

            logger.info('시뮬레이션 시작. "시뮬레이션 완료"가 표시될 때까지 종료하지 말아주세요.')
            logger.info('processors: {}', processors)

            runIDFs(self._idf_iterator(), processors=processors)

            logger.info('시뮬레이션 완료')
        else:
            logger.info('변경된 idf 파일을 저장하고 시뮬레이션은 시행하지 않습니다.')
            for _ in self._idf_iterator():
                pass
            logger.info('저장 완료')

    def _summarize(self, table: str):
        csvs = list(self._paths['output'].glob('*-table.csv'))
        if not csvs:
            raise FileNotFoundError('결과 파일이 없습니다.')

        dfs = []
        for csv in csvs:
            df = read_table_csv(path=csv, table=table)
            df['case'] = csv.name.rstrip('-table.csv')
            dfs.append(df)

        summ: pd.DataFrame = pd.concat(dfs)
        summ = summ[['case'] + [x for x in summ.columns if x != 'case']]

        path = self._paths['output'].joinpath(f'[summary] {table}.csv')
        summ.to_csv(path, encoding='utf-8-sig', index=False)
        logger.info('summary saved: "{}"', path)

    def summarize(self):
        for table in track(self._option['summary'],
                           description='Summarizing...'):
            try:
                self._summarize(table)
            except ValueError as e:
                logger.error(e)

        logger.info('Summarizing 완료')
