"""서기연 GR 사용량 분석"""

import dataclasses as dc
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


def _schedule(text: str) -> list:
    return [x.strip() for x in text.rstrip(';').split(',')]


class GRCase(EnergyPlusCase):

    def _set_cop(self, target: str, cop: float):
        # [deprecated] Coil:Cooling:DX:SingleSpeed 대신
        # AirConditioner:VariableRefrigerantFlow 사용
        if target not in ('Cooling', 'Heating'):
            raise ValueError(f'target not in ("Cooling", "Heating"): {target}')

        objs = self._get_objs(f'Coil:{target}:DX:SingleSpeed')
        for obj in objs:
            attr = f'Gross_Rated_{target}_COP'
            assert hasattr(obj, attr)
            setattr(obj, attr, cop)

    def set_cop(self, cooling: float, heating: float):
        objs = self._get_objs('AirConditioner:VariableRefrigerantFlow')
        assert len(objs) == 1

        obj = objs[0]
        obj.Gross_Rated_Cooling_COP = cooling
        obj.Gross_Rated_Heating_COP = heating

    def set_fcu(self, cooling: float, heating: float):
        # cooling
        chillers = self._get_objs('Chiller:Electric:EIR')
        assert len(chillers) == 1
        chiller = chillers[0]
        chiller.Reference_COP = cooling

        # heating
        boilers = self._get_objs('Boiler:HotWater')
        assert len(boilers) == 1
        boilder = boilers[0]
        boilder.Nominal_Thermal_Efficiency = heating

    def set_water_heater_effectiveness(self, value: float):
        objs = self._get_objs('WaterHeater:Mixed')
        for obj in objs:
            obj.Use_Side_Effectiveness = value

    def _temperature_schedule(self, name: str, type_limits='Temperature'):
        return (x for x in self._get_objs('Schedule:Compact')
                if x.obj[2] == type_limits and name in x.obj[1])

    def _set_temperature_schedule(self, name, schedule: list):
        objs = self._temperature_schedule(name=name)
        for obj in objs:
            obj.obj = obj.obj[:3] + schedule

    def set_temperature_schedule(self, cooling, heating):
        self._set_temperature_schedule(name='Cooling Setpoint',
                                       schedule=cooling)
        self._set_temperature_schedule(name='Heating Setpoint',
                                       schedule=heating)


@dc.dataclass
class _Condition:
    idf: Path
    year: int
    occupancy: float
    lighting_level: float
    water_heater_effectiveness: float
    cop: list
    fcu: list
    schedule: dict  # keys: name, cooling, heating


@dc.dataclass
class _Conditions:
    idf: list
    year: list
    occupancy: list
    lighting_level: list
    water_heater_effectiveness: list
    cop: list
    fcu: list
    schedule: list

    _total: int = dc.field(init=False)

    @property
    def total(self):
        return self._total

    def _variables(self):
        return (getattr(self, x.name)
                for x in dc.fields(self)
                if x.name != '_total')

    @staticmethod
    def _prep_schedule(schedule: dict):
        schedule['cooling'] = _schedule(schedule['cooling'])
        schedule['heating'] = _schedule(schedule['heating'])

        return schedule

    def __post_init__(self):
        lengths = (len(x) for x in self._variables())
        self._total = reduce(lambda x, y: x * y, lengths, 1)

        self.schedule = [self._prep_schedule(x) for x in self.schedule]

    def iter(self):
        for x in track(product(*self._variables()), total=self.total):
            yield _Condition(*x)


class GRRunner:
    PARAM2NAME = ('{idf}_yr{year}_oc{occupancy}_'
                  'LL{lighting_level}_WHE{water_heater_effectiveness}_'
                  'COP{cop}_FCU{fcu}_sch-{schedule}')
    NAME2PARAM = re.compile(r'^(.*?)_yr(\d+)_oc([\d\.]+)_'
                            r'LL([\d\.]+)_WHE([\d\.]+)_'
                            r'COP\[([\d\.]+),([\d\.]+)\]_'
                            r'FCU\[([\d\.]+),([\d\.]+)\]_sch-(\w+)')

    COLUMNS = ('case', 'idf', 'year', 'occupancy', 'lighting_level',
               'water_heater_effectiveness', 'cooling_cop', 'heating_cop',
               'cooling_fcu', 'heating_fcu', 'schedule')

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

    @staticmethod
    def _list_param_name(param):
        if not param:
            return '[0,0]'

        return str(param).replace(' ', '')

    def param2name(self, condition: _Condition):
        con = dc.asdict(condition)
        con.update({
            'idf': condition.idf.stem,
            'cop': self._list_param_name(condition.cop),
            'fcu': self._list_param_name(condition.fcu),
            'schedule': condition.schedule['name']
        })

        return self.PARAM2NAME.format_map(con)

    def name2param(self, name: str):
        match = self.NAME2PARAM.match(name)
        if not match:
            raise ValueError(f'case name eror: {name}')

        groups = match.groups()
        params = {x: groups[i] for i, x in enumerate(self.COLUMNS[1:])}

        return params

    def change_year(self, case: EnergyPlusCase, year: int):
        try:
            row = self._material.loc[year, :]
        except KeyError as e:
            raise ValueError(f'연도 설정 오류: {year}') from e

        case.set_year(materials={x: row[x] for x in self._material_names},
                      u_value=row[self.UVALUE])

        return case

    def _conditions(self):
        variables = self._option['case']
        variables['idf'] = self._idfs

        for key in variables:
            if not variables[key]:
                variables[key] = [None]

        return _Conditions(**variables)

    def _idf_iterator(self):
        outdir: Path = self._paths['output']

        option = dict(output_directory=outdir.as_posix(),
                      output_suffix='D',
                      verbose=self._option['EP']['verbose'],
                      readvars=self._option['EP']['readvars'])

        conditions = self._conditions()
        if conditions.total > 1:
            logger.info('total {} cases', conditions.total)

        for con in conditions.iter():
            logger.debug(con)

            case = self.case(idf=con.idf)

            if con.year:
                self.change_year(case=case, year=con.year)

            if con.occupancy:
                case.set_occupancy(con.occupancy)

            if con.lighting_level:
                case.set_lighting_level(con.lighting_level)

            if con.water_heater_effectiveness:
                case.set_water_heater_effectiveness(
                    con.water_heater_effectiveness)

            if con.cop:
                case.set_cop(cooling=con.cop[0], heating=con.cop[1])

            if con.fcu:
                case.set_fcu(cooling=con.fcu[0], heating=con.fcu[1])

            if con.schedule:
                case.set_temperature_schedule(cooling=con.schedule['cooling'],
                                              heating=con.schedule['heating'])

            name = self.param2name(con)
            case.idf.saveas(outdir.joinpath(f'{name}.idf'))
            version = '-'.join(case.idf_version()[:3])

            opt = option.copy()
            opt['output_prefix'] = name
            opt['ep_version'] = version

            yield case.idf, opt

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
        summ = summ[list(self.COLUMNS) +
                    [x for x in summ.columns if x not in self.COLUMNS]]

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
