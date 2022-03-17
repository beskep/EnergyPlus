"""서기연 GR 사용량 분석"""

from functools import reduce
from itertools import product
from pathlib import Path
import re

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
        p = root.joinpath(path).resolve()
        if p.exists():
            path = p

    return path


class GRCase(EnergyPlusCase):

    def _change_cop(self, target: str, cop: float):
        if target not in ('Cooling', 'Heating'):
            raise ValueError(f'target not in ("Cooling", "Heating"): {target}')

        objs, _ = self._get_obj_and_name(f'Coil:{target}:DX:SingleSpeed')
        for obj in objs:
            attr = f'Gross_Rated_{target}_COP'
            assert hasattr(obj, attr)
            setattr(obj, attr, cop)

    def change_cop(self, cooling: float, heating: float):
        self._change_cop(target='Cooling', cop=cooling)
        self._change_cop(target='Heating', cop=heating)


def _idf_paths(idf, root: Path):
    idf = _path(idf, root=root)

    if not idf:
        raise FileNotFoundError(idf)
    if idf.is_file():
        idfs: tuple = (idf,)
    elif idf.is_dir():
        idfs = tuple(idf.glob('*.idf'))
        if not idfs:
            raise FileNotFoundError(f'대상 폴더에 idf 파일이 없습니다: {idf}')
    else:
        raise OSError(f'IDF path error: {idf}')

    return idfs


def _other_paths(option: dict, root: Path):
    paths = {k: _path(v, root) for k, v in option['path'].items()}
    for k, v in paths.items():
        if k != 'output_template' and not (v and v.exists()):
            raise FileNotFoundError(f'{k} not found: {v}')
        logger.debug('{}: "{}"', k, v)

    return paths


class GRRunner:
    PARAM2NAME = ('{idf}_year{year}_occu{occupancy}_'
                  'lighting{lighting}_CCOP{ccop}_HCOP{hcop}')
    NAME2PARAM = re.compile(r'^(.*?)_year(\d+)_occu([\d\.]+)_'
                            r'lighting([\d\.]+)_CCOP([\d\.]+)_HCOP([\d\.]+)')
    COLS = ('case', 'idf', 'year', 'occupancy', 'lighting_level', 'cooling_cop',
            'heating_cop')
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

        for x in self._option['remove_objs'] or []:
            case.remove_obj(x)

        return case

    def param2name(self, idf: str, year: int, occupancy: float, lighting: float,
                   ccop: float, hcop: float):
        return self.PARAM2NAME.format(idf=idf,
                                      year=year,
                                      occupancy=occupancy,
                                      lighting=lighting,
                                      ccop=ccop,
                                      hcop=hcop)

    def name2param(self, name: str):
        match = self.NAME2PARAM.match(name)
        if not match:
            raise ValueError(f'case name eror: {name}')

        return dict(idf=match.group(1),
                    year=int(match.group(2)),
                    occupancy=float(match.group(3)),
                    lighting_level=float(match.group(4)),
                    cooling_cop=float(match.group(5)),
                    heating_cop=float(match.group(6)))

    def change_year(self, case: EnergyPlusCase, year: int):
        try:
            row = self._material.loc[year, :]
        except KeyError as e:
            raise ValueError(f'연도 설정 오류: {year}') from e

        case.change_year(materials={x: row[x] for x in self._material_names},
                         u_value=row[self.UVALUE])

        return case

    def _case_iterator(self):
        variables = [self._idfs]
        variables.extend(
            self._option['case'][x]
            for x in ['year', 'occupancy', 'lighting_level', 'COP'])

        total = reduce(lambda x, y: x * y, (len(x) for x in variables), 1)
        it = track(product(*variables), total=total)

        return it, total

    def _idf_iterator(self):
        last_idf = case = None
        outdir: Path = self._paths['output']

        it, total = self._case_iterator()
        if total > 1:
            logger.info('total {} cases', total)

        for idf, yr, oc, ll, cop in it:
            if last_idf != idf:
                case = self.case(idf=idf)

            self.change_year(case=case, year=yr)
            case.change_occupancy(oc)
            case.change_lighting_level(ll)

            name = self.param2name(idf=idf.stem,
                                   year=yr,
                                   occupancy=oc,
                                   lighting=ll,
                                   ccop=cop[0],
                                   hcop=cop[1])
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

    def _read_table_csv(self, path: Path, table: str):
        df = read_table_csv(path=path, table=table)

        name = path.name.rstrip('-table.csv')  # case name
        df['case'] = name

        param = self.name2param(name)
        for k, v in param.items():
            df[k] = v

        return df

    def _summarize(self, table: str):
        csvs = list(self._paths['output'].glob('*-table.csv'))
        if not csvs:
            raise FileNotFoundError('결과 파일이 없습니다.')

        summ: pd.DataFrame = pd.concat(
            (self._read_table_csv(x, table=table) for x in csvs))

        # 열 순서 변경
        summ = summ[list(self.COLS) +
                    [x for x in summ.columns if x not in self.COLS]]

        # 저장
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
