"""
서기연 GR 사용량 분석
변수 랜덤 선정
"""
from calendar import month_name
import csv
import dataclasses as dc
from pathlib import Path
import re

from eppy.modeleditor import IDF
from eppy.runner.run_functions import runIDFs
from loguru import logger
import numpy as np
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


def _paths(option: dict, root: Path):
    paths = {k: _path(v, root) for k, v in option['path'].items()}
    for k, v in paths.items():
        if k != 'output_template' and not (v and v.exists()):
            raise FileNotFoundError(f'{k} not found: {v}')
        logger.debug('{}: "{}"', k, v)

    return paths


class GRCase(EnergyPlusCase):
    # GR 스케줄 이름 (e.g. W9t18_H0t0)
    P_SCHEDULE = re.compile(r'^W\d+t\d+_H\d+t\d+')

    # Zone 스케줄 이름
    # (e.g. Schedule:Week:Daily, Block111:Zone1 Heating Availability Sch_Apr)
    P_SWD_ZONE = re.compile(r'^Block\d+:Zone\d+ (.*)$')

    def set_conditioning_temperature(self, t0: float, t1: float):
        suffix = f'_{t0}'
        schedules = [
            x for x in self._get_objs('Schedule:Day:Interval')
            if x.Name.endswith(suffix)
        ]
        if not schedules:
            raise ValueError(f'설정 온도 {t0}인 스케줄이 발견되지 않음.')

        for schedule in schedules:
            for idx in range(1, 145):
                attr = f'Value_Until_Time_{idx}'
                temperature = getattr(schedule, attr)

                if temperature == '':
                    break

                if temperature == t0:
                    setattr(schedule, attr, t1)

    def _get_reference_schedule(self, name: str, reference_schedules: list):
        match = self.P_SWD_ZONE.match(name)
        if not match:
            return None

        stype = match.group(1)

        try:
            reference = next(
                x for x in reference_schedules if x.Name.endswith(stype))
        except StopIteration as e:
            raise ValueError(f'Reference IDF에서 스케줄 {name}를 찾지 못했습니다.') from e

        return reference

    def _copy_daily_schedule(self, ridf: IDF):
        SDI = 'Schedule:Day:Interval'
        SWD = 'Schedule:Week:Daily'

        # Schedule:Day:Interval 복사
        sdi_list = self._get_objs(SDI)
        sdi_names = {x.Name for x in sdi_list}
        for rsdi in ridf.idfobjects[SDI]:
            if rsdi.Name not in sdi_names:
                sdi_list.append(rsdi)

        # Schedule:Week:Daily (Zone) 설정
        # e.g. `Schedule:Week:Daily, Block1:Zone1 Cooling SP Sch_Jun`
        rswd_list = [
            x for x in ridf.idfobjects[SWD] if self.P_SWD_ZONE.match(x.Name)
        ]
        for swd in self._get_objs(SWD):
            rswd = self._get_reference_schedule(name=swd.Name,
                                                reference_schedules=rswd_list)
            if rswd is not None:
                swd.obj = swd.obj[:2] + rswd.obj[2:]

    def _copy_work_schedule(self, ridf: IDF):
        SWD = 'Schedule:Week:Daily'
        SY = 'Schedule:Year'

        # Schedule:Week:Daily (W9t18_H0t0 형식) 복사
        swd_list = self._get_objs(SWD)
        swd_names = {x.Name for x in swd_list if self.P_SCHEDULE.match(x.Name)}
        for rswd in ridf.idfobjects[SWD]:
            if (self.P_SCHEDULE.match(rswd.Name) and
                    rswd.Name not in swd_names):
                swd_list.append(rswd)

        # Schedule:Year 복사
        # e.g. `Schedule:Year, W9t18_H0t0, Any number`
        rsy = [x for x in ridf.idfobjects[SY] if self.P_SCHEDULE.match(x.Name)]
        assert len(rsy) == 1

        self._get_objs(SY).append(rsy[0])

        return rsy[0].Name

    def set_schedule(self, reference_idf: str):
        ridf = IDF(idfname=reference_idf)

        self._copy_daily_schedule(ridf=ridf)
        schedule_name = self._copy_work_schedule(ridf=ridf)

        # 각 스케줄 지정
        for light in self._get_objs('Lights'):
            light.Schedule_Name = schedule_name

        for equipment in self._get_objs('ElectricEquipment'):
            equipment.Schedule_Name = schedule_name

        for zv in self._get_objs('ZoneVentilation:DesignFlowRate'):
            zv.Schedule_Name = schedule_name

        for people in self._get_objs('PEOPLE'):
            people.Number_of_People_Schedule_Name = schedule_name


@dc.dataclass
class _FileCondition:
    area: int
    wwr: int  # window to wall ratio
    aspect_ratio: int

    def filename(self):
        return f'{self.area}-{self.wwr}-1_{self.aspect_ratio}'

    def asdict(self):
        return dc.asdict(self)


@dc.dataclass
class _NumericalCondition:
    shgc: float
    lighting_level: float
    occupancy: float
    equipment_power: float
    cooling_temperature: float
    heating_temperature: float
    material_thickness: dict
    ct0: dict  # original conditioning temperature

    def __post_init__(self):
        self.cooling_temperature = np.round(self.cooling_temperature, 1)
        self.heating_temperature = np.round(self.heating_temperature, 1)

    def set_case(self, case: GRCase):
        case.set_simple_glazing_system(shgc=self.shgc)
        case.set_occupancy(self.occupancy)
        case.set_lighting_level(self.lighting_level)
        case.set_equipment_power(self.equipment_power)

        case.set_conditioning_temperature(t0=self.ct0['cooling'],
                                          t1=self.cooling_temperature)
        case.set_conditioning_temperature(t0=self.ct0['heating'],
                                          t1=self.heating_temperature)

        for m, t in self.material_thickness.items():
            case.set_material_thickness(material=m, thickness=t)

    def asdict(self):
        d = dc.asdict(self)
        mt = d.pop('material_thickness')
        d.pop('ct0')
        d.update({f'thickness_{k}': v for k, v in mt.items()})

        return d


@dc.dataclass
class _CategoricalCondition:
    north_axis: float  # [deg]
    schedule: str
    base_schedule: str
    schedule_dir: Path

    def set_case(self, case: GRCase):
        case.set_north_axis(self.north_axis)

        if self.schedule != self.base_schedule:
            reference = self.schedule_dir.joinpath(f'{self.schedule}.idf')
            case.set_schedule(reference_idf=reference.as_posix())

    def asdict(self):
        return {'north_axis': self.north_axis, 'schedule': self.schedule}


@dc.dataclass
class _Condition:
    file: _FileCondition
    numerical: _NumericalCondition
    categorical: _CategoricalCondition

    def asdict(self):
        d = self.file.asdict()
        d.update(self.numerical.asdict())
        d.update(self.categorical.asdict())

        return d

    def set_case(self, case: GRCase):
        self.categorical.set_case(case)
        self.numerical.set_case(case)


class _ConditionGenerator:

    def __init__(self, option: dict, seed=None) -> None:
        self.option = option

        NUM, MT = 'numerical', 'material_thickness'
        self.params = {k: sorted(v) for k, v in option[NUM].items() if k != MT}
        self.thickness = option[NUM][MT].copy()

        self.schedule_dir = Path(option['misc']['schedule_dir']).resolve()
        self.schedule_dir.stat()

        self.rng = np.random.default_rng(seed)

    def _choice(self, option: dict):
        return {k: self.rng.choice(v) for k, v in option.items()}

    def generate_condition(self):
        params = {
            k: np.round(self.rng.uniform(*v), 4)
            for k, v in self.params.items()
        }
        thickness = {
            k: np.round(self.rng.uniform(*v), 4)
            for k, v in self.thickness.items()
        }

        fc = _FileCondition(**self._choice(self.option['file']))
        nc = _NumericalCondition(
            ct0=self.option['misc']['original_conditioning_temperature'],
            material_thickness=thickness,
            **params)
        cc = _CategoricalCondition(
            base_schedule=self.option['misc']['base_schedule'],
            schedule_dir=self.schedule_dir,
            **self._choice(self.option['categorical']))

        return _Condition(file=fc, numerical=nc, categorical=cc)


class GRRunner:
    MN = list(month_name)
    P_SCHEDULE = re.compile(r'^W(\d+t\d+)_H(\d+t\d+)')

    def __init__(self, case: StrPath, seed=None) -> None:
        case = Path(case)
        root = case.parent

        with case.open('r', encoding='utf-8') as f:
            option = yaml.safe_load(f)

        self._option = option
        self._paths = _paths(option=option, root=root)
        self._vars: dict = self._option['summary']['variable']

        self._generator = _ConditionGenerator(self._option['case'], seed=seed)

    def case(self, idf):
        case = GRCase(idd=self._paths['idd'],
                      idf=idf,
                      epw=self._paths['epw'],
                      output_template=self._paths['output_template'])

        for x in self._option['remove_objs'] or []:
            case.remove_obj(x)

        case.idf.removeallidfobjects('Output:Meter')

        for ov in case._get_objs('Output:Variable'):
            if ov.Reporting_Frequency in ('hourly', 'daily'):
                case.idf.removeidfobject(ov)

        summary_reports = case._get_objs('Output:Table:SummaryReports')
        summary_reports[0].obj = [
            'Output:Table:SummaryReports', 'AllSummary', 'AllMonthly'
        ]

        return case

    def _get_paths(self):
        idfdir: Path = Path(self._option['case']['misc']['idf_dir']).resolve()
        outdir: Path = self._paths['output']

        idfdir.stat()
        outdir.stat()

        csv_path = outdir.joinpath('case.csv')
        if csv_path.exists():
            raise FileExistsError(f'기존 시뮬레이션 결과가 존재함: "{csv_path}"')

        return idfdir, outdir, csv_path

    def _idf_iterator(self, size: int):
        with logger.catch(FileExistsError, reraise=True):
            idfdir, outdir, csv_path = self._get_paths()

        option = dict(output_directory=outdir.as_posix(),
                      output_suffix='D',
                      verbose=self._option['EP']['verbose'],
                      readvars=self._option['EP']['readvars'])

        with csv_path.open('w', encoding='utf-8', newline='') as cf:
            condition = self._generator.generate_condition()
            fields = ['index', 'name'] + sorted(condition.asdict())
            writer = csv.DictWriter(cf, fieldnames=fields)
            writer.writeheader()

            for idx in track(range(size)):
                condition = self._generator.generate_condition()

                idf = idfdir.joinpath(f'{condition.file.filename()}.idf')
                assert idf.exists()
                case = self.case(idf=idf)

                condition.set_case(case)

                name = f'case{idx:06d}'
                case.idf.saveas(outdir.joinpath(f'{name}.idf'))

                opt = option.copy()
                opt['output_prefix'] = name
                opt['ep_version'] = '-'.join(case.idf_version()[:3])

                writer.writerow({
                    'index': idx,
                    'name': name,
                    **condition.asdict()
                })

                yield case.idf, opt

    def run(self, size: int, run=True):
        if run:
            processors = int(self._option['EP']['processors'])

            logger.info('시뮬레이션 시작. "시뮬레이션 완료"가 표시될 때까지 종료하지 말아주세요.')
            logger.info('processors: {}', processors)

            runIDFs(self._idf_iterator(size=size), processors=processors)

            logger.info('시뮬레이션 완료')
        else:
            logger.info('변경된 idf 파일을 저장하고 시뮬레이션은 시행하지 않습니다.')

            for _ in self._idf_iterator(size=size):
                pass

            logger.info('저장 완료')

    def _read_table_csv(self, path: Path, table: str):
        df = read_table_csv(path=path, table=table)

        df = df.loc[[x in self._vars for x in df['variable']], :]
        df['variable'] = [self._vars[x] for x in df['variable']]
        df['month'] = [self.MN.index(x) for x in df['index']]
        df['case'] = path.name.rstrip('-table.csv')

        return df[['case', 'month', 'variable', 'value']]

    def _iter_tables(self, table):
        csvs = list(self._paths['output'].glob('*-table.csv'))
        if not csvs:
            raise FileNotFoundError('결과 파일이 없습니다.')

        for csv in track(csvs):
            try:
                df = self._read_table_csv(csv, table=table)
            except ValueError:
                logger.warning('결과 table 파일 오류: "{}"', csv)
            else:
                yield df

    def _summarize_table(self, table: str):
        csvs = list(self._paths['output'].glob('*-table.csv'))
        if not csvs:
            raise FileNotFoundError('결과 파일이 없습니다.')

        df: pd.DataFrame = pd.concat(self._iter_tables(table))

        fname = table.replace(':', '').replace(',', '')
        path = self._paths['output'].joinpath(f'[summary] {fname}.csv')
        df.to_csv(path, encoding='utf-8-sig', index=False)
        logger.info('summary saved: "{}"', path)

        return df

    def _working_hours(self, schedule: str, schedule_time: dict):
        m = self.P_SCHEDULE.match(schedule)

        try:
            assert m is not None
            weekday = schedule_time[m.group(1)]
            weekend = schedule_time[m.group(2)]
        except (AssertionError, IndexError) as e:
            raise ValueError(f'스케줄 형식 오류: "{schedule}"') from e

        return weekday, weekend

    def _u_value(self, material: dict, thickness: np.ndarray):
        uinv = thickness / material['conductivity']
        uinv_others = np.divide(material['other_thickness'],
                                material['other_conductivity'])

        return np.divide(1.0, uinv + np.sum(uinv_others))

    def _summarize_post(self, df: pd.DataFrame):
        # WWR
        df['wwr'] /= 100.0

        # round lighting level
        df['lighting_level'] = np.round(df['lighting_level'], 2)

        # working hours
        schedule_time = self._option['case']['misc']['working_hours']
        working_hours = np.array(
            [self._working_hours(x, schedule_time) for x in df['schedule']])

        df[['working_hours_weekday', 'working_hours_weekend']] = working_hours

        # u value
        for name, material in self._option['case']['misc']['material'].items():
            thickness = df[f'thickness_{name}'].values.flatten()
            loc = material['location']
            df[f'u_{loc}'] = np.round(self._u_value(material, thickness), 4)

        # north axis
        nad = self._option['case']['misc']['north_axis']
        df['north_axis_kor'] = [nad[x] for x in df['north_axis']]

        # sort
        col_index = ['case', 'month']
        col_load = [x for x in df.columns if x.startswith('load_')]
        cols = set(col_index + col_load)
        col_input = sorted(
            x for x in df.columns
            if (x not in cols) and not x.startswith('thickness_'))

        return df[col_index + col_load + col_input]

    @staticmethod
    def _summarize_efficiency(df: pd.DataFrame, efficiency: dict):
        # 같은 케이스에 같은 효율 랜덤 생성
        dfe = df[['case']].drop_duplicates()
        dfe['efficiency_cooling'] = np.round(
            np.random.uniform(*efficiency['cooling'], size=dfe.shape[0]), 2)
        dfe['efficiency_heating'] = np.round(
            np.random.uniform(*efficiency['heating'], size=dfe.shape[0]), 2)

        df = df.merge(dfe, how='left', left_on='case', right_on='case')

        df['load_cooling'] /= df['efficiency_cooling']
        df['load_heating'] /= df['efficiency_heating']

        return df

    def _save_summary(self, df: pd.DataFrame, name: str):
        outdir = self._paths['output']

        # monthly
        df.to_csv(outdir.joinpath(f'[summary] {name}_monthly.csv'),
                  index=False,
                  encoding='utf-8-sig')

        cols_load = [x for x in df.columns if x.startswith('load_')]
        df_input = df.loc[[x == 1 for x in df['month']],
                          [not x.startswith('load_') for x in df.columns]]
        df_input = df_input.drop(columns='month').set_index('case')

        # annual
        df_annual = df[['case'] + cols_load].groupby('case').sum()
        df_annual = df_annual.join(df_input)
        df_annual.to_csv(outdir.joinpath(f'[summary] {name}_annual.csv'),
                         encoding='utf-8-sig')

        # monthly (pivot)
        df_load = pd.melt(df[['case', 'month'] + cols_load],
                          id_vars=['case', 'month'])
        df_load['variable'] = [
            f'{v}_{m:02d}'.replace('load_', '')
            for v, m in zip(df_load['variable'], df_load['month'])
        ]
        dfp = pd.pivot(df_load.drop(columns='month'),
                       index='case',
                       columns='variable',
                       values='value')
        dfp = dfp.join(df_input)
        dfp.to_csv(outdir.joinpath(f'[summary] {name}_monthly_pivot.csv'),
                   encoding='utf-8-sig')

    def summarize(self):
        dfs = []
        for table in self._option['summary']['table']:
            logger.info('summarize "{}"', table)
            dfs.append(self._summarize_table(table))

        input_vars = pd.read_csv(self._paths['output'].joinpath('case.csv'))
        input_vars = input_vars.drop(columns='index').rename(
            columns={'name': 'case'})

        df: pd.DataFrame = pd.concat(dfs)
        dfp = pd.pivot(df,
                       index=['case', 'month'],
                       columns='variable',
                       values='value').reset_index()
        dfm = dfp.merge(input_vars, how='left', left_on='case', right_on='case')

        dfm = self._summarize_post(dfm)
        self._save_summary(dfm, name='original')

        # EHP, 중앙식
        ef = self._option['case']['misc']['efficiency']
        self._save_summary(self._summarize_efficiency(dfm.copy(), ef['EHP']),
                           name='EHP')
        self._save_summary(self._summarize_efficiency(dfm.copy(),
                                                      ef['central']),
                           name='central')

        logger.info('Summarizing 완료')
