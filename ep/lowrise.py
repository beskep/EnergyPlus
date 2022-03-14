from itertools import chain
from itertools import product
import os
import re
from warnings import warn

from eppy.bunch_subclass import EpBunch
import numpy as np
import pandas as pd
from tqdm import tqdm

from ep.ep import EnergyPlusCase
from ep.ep import read_bunches


class CaseSetting:

    def __init__(self, variables, material):
        self.variable: dict = variables
        self.material: pd.DataFrame = material

    @property
    def case_count(self):
        return np.prod([len(x) for x in self.variable.values()])

    def get_case_iterator(self):
        return product(*self.variable.values())

    def get_material_setting(self, year, loc):
        result: pd.DataFrame = self.material.loc[
            np.logical_and(self.material['year'] ==
                           year, self.material['location'] == loc), :]
        assert result.shape[0] == 1
        return result.reset_index(drop=True)


def run_energy_plus(idd_path: str,
                    idf_path: str,
                    variables: dict,
                    epw_path: dict,
                    material: pd.DataFrame,
                    infiltration: dict,
                    schedule: dict,
                    heat_metabolic: dict,
                    heat_equipment: dict,
                    output_template=None,
                    remove_list=None,
                    save_idf=False,
                    verbose=False):
    """버려"""
    print('\n\nUSE AT YOUR OWN RISK!\n')

    case = EnergyPlusCase(idd_path,
                          idf_path,
                          '',
                          output_template=output_template)

    if remove_list:
        for variable in remove_list:
            case.remove_obj(variable)

    if output_template:
        case.set_output()

    output_path = os.path.abspath('./result/')
    verbose_ep = 'v' if verbose else 'q'

    case_setting = CaseSetting(variables, material)

    case_iter = case_setting.get_case_iterator()
    for year, loc, azimuth, people, schedule_percent in tqdm(case_iter):
        case.change_north_axis(azimuth)
        case.change_people_density(heat_metabolic[people])
        case.change_equipment_power(heat_equipment[people])
        case.change_schedule(schedule[schedule_percent])
        case.change_infiltration(infiltration[year])

        # 재료, 창문 u-value 변경
        df_material = case_setting.get_material_setting(year, loc)
        materials_list = [
            x for x in df_material.columns
            if x not in ['year', 'location', 'window']
        ]
        case.change_window_u_value(df_material.loc[0, 'window'])
        for m in materials_list:
            case.change_thickness(m, df_material.loc[0, m])

        # 저장 파일 이름
        case_name = 'year{}_loc_{}_azi{}_people{}_sche{:.2f}'.format(
            year, loc, azimuth, people, schedule_percent)

        # idf 저장
        if save_idf:
            case.idf.saveas(os.path.join(output_path, case_name + '.idf'))

        try:
            case.idf.run(weather=epw_path[loc],
                         output_directory=output_path,
                         output_prefix='eplus_{}_'.format(case_name),
                         readvars=True,
                         verbose=verbose_ep)
        except Exception as e:
            with open('./error.txt', 'a') as f:
                f.write('{}: {}\n'.format(case_name, e))

        if verbose:
            print()

    return


def apply_blind(case: EnergyPlusCase, blind, interior: bool):
    """
    사용하기 전에 대상 케이스와 복사할 케이스의
    WindowMaterial:SimpleGlazingSystem가 일치하는지 확인할 것!

    Parameters
    ----------
    case : EnergyPlusCase
        변경 대상 케이스
    blind : [type]
        붙여넣기 할 블라인드 정보 모음
    interior : bool
        안/밖
    """

    # shading_control = [
    #   x for x in blind if x.obj[0].lower() == 'windowproperty:shadingcontrol']
    # assert len(shading_control) == 1
    #
    # if not interior:
    #   construction = [x for x in blind if x.obj[0].lower() == 'construction']
    #   assert len(construction) == 1
    #   construction[0].obj[2:4] = construction[0].obj[-1:-3:-1]
    #
    #   shading_control[0].Shading_Type = 'ExteriorBlind'

    for x in blind:
        case.idf.copyidfobject(x)

    shading_control = case.idf.idfobjects['WindowProperty:ShadingControl']
    assert len(shading_control) == 1
    if not interior:
        shading_control[0].Shading_Type = 'ExteriorBlind'

        construction_name = [
            x for x in blind if x.obj[0].lower() == 'construction'
        ][0].Name
        construction = case.idf.idfobjects['construction']
        construction = [x for x in construction if x.Name == construction_name]
        assert len(construction) == 1
        construction[0].obj[2:4] = construction[0].obj[-1:-3:-1]

    fsd = case.idf.idfobjects['FenestrationSurface:Detailed']
    for f in fsd:
        f.Shading_Control_Name = shading_control[0].Name

    return


def change_lights_power(case: EnergyPlusCase, multiply=1.0):
    """
    모든 light의 Lighting Level을 바꿈
    LED 적용하려면 r=0.7 입력

    Parameters
    ----------
    case : EnergyPlusCase
        변경 대상 케이스
    multiply : float, optional
        에너지 소비량 변화율
    """
    if multiply == 1.0:
        warn('조명 에너지 소비량이 변하지 않습니다 (r=1.0)')
        return

    lights = case.idf.idfobjects['LIGHTS']
    for light in lights:
        light.Lighting_Level *= multiply
    return


def add_inner_material(case: EnergyPlusCase,
                       target_wall: str,
                       material: EpBunch,
                       thickness=0.001):
    if thickness:
        material.Thickness = thickness
    case.idf.copyidfobject(material)

    p = re.compile(target_wall)
    walls = case.idf.idfobjects['construction']
    walls = [x for x in walls if p.match(x.Name)]

    assert len(walls) <= 2

    for wall in walls:
        if not wall.Name.endswith('_Rev'):
            # reversed 벽이 아닐 때: 마지막 재료 (내부)에 대상 재료를 추가
            wall.obj.append(material.Name)
        else:
            # 제일 바깥에 추가
            assert len(wall.obj) <= 10
            wall.obj.insert(2, material.Name)


class LowriseEPRunner:
    infiltration = {1980: 1.0, 1987: 1.0, 2001: 0.75, 2007: 0.5}
    heat_metabolic = {2: 0.0338, 3: 0.0508, 4: 0.0677}
    heat_equipment = {2: 1.94, 3: 2.17, 4: 2.40}

    existing_variables = ['year', 'loc', 'azimuth', 'people', 'schedule']
    energy_variables = ['blind', 'led', 'window', 'insulation']

    def __init__(self,
                 idd_path,
                 base_idf_path,
                 dim_idf_path,
                 material_path,
                 window_path,
                 output_template=None,
                 remove_list=None):
        self._idd = idd_path
        self._base_idf = base_idf_path
        self._dim_idf = dim_idf_path
        self._output_template = output_template
        self._remove_list = remove_list

        self._material = pd.read_csv(material_path)
        self._materials_name = [
            x for x in self._material.columns
            if x not in ['year', 'location', 'tech']
        ]
        self._window = pd.read_csv(window_path)

        self._base_case = None
        self._dim_case = None

    def reset(self):
        self._base_case = EnergyPlusCase(self._idd, self._base_idf, '',
                                         self._output_template)
        self._dim_case = EnergyPlusCase(self._idd, self._dim_idf, '',
                                        self._output_template)

        if self._remove_list:
            for x in self._remove_list:
                self._base_case.remove_obj(x)
                self._dim_case.remove_obj(x)

        if self._output_template:
            self._base_case.set_output()
            self._dim_case.set_output()

    @staticmethod
    def _get_condition(row, condition):
        x = getattr(row, condition)
        if isinstance(x, str):
            x = x.lower()
        return x

    def change_existing_variable(self, case, row, metabolic_schedule_id):
        year, loc, azimuth, people, schedule = [
            self._get_condition(row, x) for x in self.existing_variables
        ]
        window, insulation = [
            self._get_condition(row, x) for x in ['window', 'insulation']
        ]
        if pd.isna(insulation) or insulation.lower().startswith('pai'):
            insulation = 'original'
        if pd.isna(window):
            window = 'original'

        case.change_north_axis(azimuth)
        case.change_people_density(self.heat_metabolic[people])
        case.change_equipment_power(self.heat_equipment[people])
        case.change_infiltration(self.infiltration[year])

        # 메타볼릭 스케줄
        schedule_list = [1.0] * 24
        schedule_list[8:18] = [schedule] * 10
        case.change_schedule(schedule_list, metabolic_schedule_id)

        # 재료 두께
        myear = self._material['year'] == year
        mloc = self._material['location'] == loc
        mtech = self._material['tech'] == insulation
        mmask = np.argwhere(myear & mloc & mtech)
        assert mmask.size == 1
        mcondition = self._material.iloc[mmask[0]]
        for x in self._materials_name:
            case.change_thickness(x, mcondition[x].values[0])

        # 창문 열관류율
        wyear = self._window['year'] == year
        wloc = self._window['location'] == loc
        wtech = self._window['tech'] == window
        wmask = np.argwhere(wyear & wloc & wtech)
        assert wmask.size == 1
        wcondition = self._window.iloc[wmask[0]]
        case.change_window_u_value(wcondition['window'].values[0])

        return case

    def change_energy_variable(self, row, blind_bunches, paint_bunch,
                               painting_wall, paint_thickness):
        blind, led, _, insulation = [
            self._get_condition(row, x) for x in self.energy_variables
        ]

        # LED 설정
        if not pd.isna(led) and led not in ['led', 'dim', 'leddim', 'dimled']:
            raise ValueError('LED 변수 잘못 입력함.')
        if isinstance(led, str) and 'dim' in led:
            case = self._dim_case
        else:
            case = self._base_case
        if isinstance(led, str) and 'led' in led:
            change_lights_power(case, 0.7)

        # 블라인드 설정
        if not pd.isna(blind):
            if blind not in ['blind', 'louver']:
                raise ValueError('블라인드 변수 잘못 입력함.')
            apply_blind(case, blind_bunches, interior=(blind == 'blind'))

        # 페인트 설정
        if not pd.isna(insulation) and insulation.startswith('pai'):
            add_inner_material(case, painting_wall, paint_bunch,
                               paint_thickness)

        return case

    def run(self,
            conditions: pd.DataFrame,
            epw: dict,
            blind_bunches=None,
            paint_bunch=None,
            metabolic_schedule_id=10013,
            painting_wall='^(K_Outdoor Original Wall 1979)(_Rev)?$',
            paint_thickness=0.001,
            run=True,
            save_idf=False,
            verbose=False):
        # metabolic schedule 번호 확인할것!
        assert all([x in conditions.columns for x in self.existing_variables])
        assert all([x in conditions.columns for x in self.energy_variables])

        conditions.sort_values(self.energy_variables, inplace=True)

        last_energy_var = None
        output_path = os.path.abspath('./result/')
        epverbose = 'v' if verbose else 'q'

        for row in tqdm(conditions.itertuples(), total=conditions.shape[0]):
            energy_var = [getattr(row, x) for x in self.energy_variables]
            case: EnergyPlusCase = None

            if last_energy_var != energy_var:
                self.reset()
                case = self.change_energy_variable(row, blind_bunches,
                                                   paint_bunch, painting_wall,
                                                   paint_thickness)
            last_energy_var = energy_var.copy()

            existing_var = [
                self._get_condition(row, x) for x in self.existing_variables
            ]

            if all([not pd.isna(x) for x in existing_var]):
                case = self.change_existing_variable(case, row,
                                                     metabolic_schedule_id)

            case_name = 'y{}_loc{}_a{}_p{}_s{}'.format(
                *[to_case_name(x) for x in existing_var])
            case_name += '_b{}_led{}_w{}_i{}'.format(
                *[to_case_name(x) for x in energy_var])

            if save_idf:
                case.idf.saveas(os.path.join(output_path, case_name + '.idf'))

            if run:
                case.idf.run(weather=epw[existing_var[1]],
                             output_directory=output_path,
                             output_prefix='eplus_{}_'.format(case_name),
                             readvars=True,
                             verbose=epverbose)


def to_case_name(x, int_fmt='02d', float_fmt='.2f'):
    if pd.isna(x):
        res = 'Na'
    elif isinstance(x, int):
        res = '{:{}}'.format(x, int_fmt)
    elif isinstance(x, float):
        res = '{:{}}'.format(x, float_fmt)
    else:
        res = str(x).title()
    return res


if __name__ == '__main__':
    idd_path = os.path.abspath('./idd/V8-3-0-Energy+.idd')
    out_template_path = os.path.abspath('./idf/output_template.idf')

    base_idf_path = os.path.abspath('./idf/39_A.idf')
    dim_idf_path = os.path.abspath('./idf/39_testfile_dim.idf')
    blind_idf_path = os.path.abspath('./idf/59_testfile_bli.idf')
    paint_idf_path = os.path.abspath('./idf/59_testfile_pai.idf')

    epw = {
        'central': os.path.abspath('./input/Hybrid Seoul-hourEPW.epw'),
        'southern': os.path.abspath('./input/Hybrid Busan-hourEPW.epw'),
        'jeju': os.path.abspath('./input/Hybrid Cheju-hourEPW.epw')
    }

    # 제거할 항목
    remove_list = [
        'OUTPUT:CONSTRUCTIONS',
        'OUTPUT:DAYLIGHTFACTORS',
        'OUTPUT:DEBUGGINGDATA',
        'OUTPUT:DIAGNOSTICS',
        'OUTPUT:ENERGYMANAGEMENTSYSTEM',
        'OUTPUT:ENVIRONMENTALIMPACTFACTORS',
        'OUTPUT:ILLUMINANCEMAP',
        'OUTPUT:METER',
        'OUTPUT:METER:CUMULATIVE',
        'OUTPUT:METER:CUMULATIVE:METERFILEONLY',
        'OUTPUT:METER:METERFILEONLY',
        'OUTPUT:PREPROCESSORMESSAGE',
        'OUTPUT:SCHEDULES',
        'OUTPUT:SURFACES:DRAWING',
        'OUTPUT:SURFACES:LIST',
        'OUTPUTCONTROL:ILLUMINANCEMAP:STYLE',
        'OUTPUTCONTROL:REPORTINGTOLERANCES',
        'OUTPUTCONTROL:SIZING:STYLE',
        'OUTPUTCONTROL:SURFACECOLORSCHEME',
        'OUTPUTCONTROL:TABLE:STYLE',
    ]

    # blind 설정 불러오기
    months = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
        'Nov', 'Dec'
    ]
    month_schedule = '^K_100_blind_({})$'.format('|'.join(
        ['(' + x + ')' for x in months]))
    blind = read_bunches(
        idd_path, blind_idf_path, {
            'schedule:day:interval': '^10018$',
            'schedule:day:list': '^10021$',
            'schedule:week:daily': month_schedule,
            'schedule:year': '^K_100_blind$',
            'WindowMaterial:blind': '^20001$',
            'construction': '^2001$',
            'WindowProperty:ShadingControl': '^1001$'
        })
    blind = list(chain.from_iterable(blind))

    # paint 설정 불러오기
    paint = read_bunches(idd_path, paint_idf_path,
                         {'material': '^INSULATION PAINT.*'})
    paint = list(chain.from_iterable(paint))
    assert len(paint) == 1
    paint = paint[0]

    conditions = pd.read_csv('./input/condition.csv')

    runner = LowriseEPRunner(idd_path,
                             base_idf_path,
                             dim_idf_path,
                             './input/material.csv',
                             './input/window.csv',
                             output_template=out_template_path,
                             remove_list=remove_list)

    runner.run(conditions,
               epw,
               blind_bunches=blind,
               paint_bunch=paint,
               metabolic_schedule_id=None,
               painting_wall='^(K_Outdoor Original Wall 1979)(_Rev)?$',
               paint_thickness=0.001,
               run=True,
               save_idf=True,
               verbose=False)
