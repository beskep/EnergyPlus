from pathlib import Path
import re
from typing import Optional

from eppy.bunch_subclass import EpBunch
from eppy.idf_msequence import Idf_MSequence
from eppy.modeleditor import IDF
from loguru import logger

from .utils import StrPath


def _path(path):
    path = Path(path).resolve()
    path.stat()

    return path.as_posix()


class EnergyPlusCase:

    def __init__(self,
                 idd: StrPath,
                 idf: StrPath,
                 epw: Optional[StrPath] = None,
                 output_template: Optional[StrPath] = None):
        idd = _path(idd)
        idf = _path(idf)
        epw = _path(epw) if epw else ''

        if IDF.getiddname() is None:
            IDF.setiddname(idd)

        self._idf = IDF(idfname=idf, epw=epw)
        self._epw = epw

        self._zone = self._idf.idfobjects['ZONE']
        self._zone_name = tuple(z.Name for z in self._zone)
        self._material = None

        self._template = IDF(str(output_template)) if output_template else None

        logger.debug('EP case init')
        logger.debug('idd: "{}"', idd)
        logger.debug('idf: "{}"', idf)
        logger.debug('epw: "{}"', epw)

    @property
    def idf(self):
        return self._idf

    def zone(self, name: str) -> EpBunch:
        try:
            idx = self._zone_name.index(name)
        except IndexError as e:
            raise IndexError(f'Zone name error: {name}') from e

        return self._zone[idx]

    def idf_version(self):
        return self.idf.idfobjects['version'][0].Version_Identifier.split('.')

    @property
    def material(self) -> Idf_MSequence:
        if self._material is None:
            self._material = self._idf.idfobjects['Material']

        return self._material

    def save(self, path):
        path = Path(path).resolve().as_posix()
        logger.debug('idf save to "{}"', path)
        self._idf.saveas(path)

    def run(self,
            output_directory: StrPath,
            output_prefix='eplus',
            output_suffix='D',
            readvars=True,
            verbose='q'):
        # TODO doc -> eppy.runner.run_functions.run
        self.idf.run(weather=self._epw,
                     output_directory=_path(output_directory),
                     output_prefix=output_prefix,
                     output_suffix=output_suffix,
                     readvars=readvars,
                     verbose=verbose)

    def _get_obj_and_name(self, obj):
        objs: Idf_MSequence = self._idf.idfobjects[obj]
        names: tuple[str, ...] = tuple(x.Name for x in objs)

        return objs, names

    def set_output(self, variable='Output:Variable'):
        self._idf.idfobjects[variable] = self._template.idfobjects[variable]

    def remove_obj(self, variable):
        obj = self._idf.idfobjects[variable]
        if len(obj):
            self._idf.removeidfobject(obj[0])

    def change_north_axis(self, north_axis):
        building = self._idf.idfobjects['BUILDING'][0]
        building.North_Axis = north_axis

    def change_infiltration(self, infiltration_rate: float):
        """
        Parameters
        ----------
        infiltration_rate : float
            [1/h]
        """
        ir = infiltration_rate / 3600.0  # [1/sec]
        objs, names = self._get_obj_and_name('ZoneInfiltration:DesignFlowRate')

        for obj, n in zip(objs, names):
            volume = self.zone(n.rstrip(' Infiltration')).Volume
            obj.Design_Flow_Rate = volume * ir

    def change_occupancy(self, density: float):
        """
        Parameters
        ----------
        density : float
            [people/m^2]
        """
        objs, names = self._get_obj_and_name('PEOPLE')

        for obj, n in zip(objs, names):
            area = self.zone(n.lstrip('People ')).Floor_Area
            obj.Number_of_People = density * area

    def change_equipment_power(self, power: float):
        """
        Parameters
        ----------
        power : float
            [W/m^2]
        """
        # FIXME 'Equipment \d' 형식이 아닐 수 있음
        pattern = re.compile(r'\sEquipment\s\d+.*$')
        objs, names = self._get_obj_and_name('ElectricEquipment')
        names = tuple(pattern.sub('', x) for x in names)

        for obj, n in zip(objs, names):
            area = self.zone(n).Floor_Area
            obj.Design_Level = power * area

    def change_lighting_level(self, power: float):
        """
        Parameters
        ----------
        power : float
            [W/m^2]
        """
        objs, names = self._get_obj_and_name('Lights')

        for obj, n in zip(objs, names):
            area = self.zone(n.rstrip(' General lighting')).Floor_Area
            obj.Lighting_Level = area * power

    def change_schedule(self, schedule: list, metabolic_schedule_id=None):
        """
        범용적으로 못 씀
        현재 lighting/metabolic 두 스케줄만 있을 때
        0시의 on/off 여부로 둘을 구분하고 있음
        """
        schedule_objs = self._idf.idfobjects['Schedule:Day:List']
        index_first_value = [s.objls.index('Value_1') for s in schedule_objs]
        if metabolic_schedule_id:
            metabolic_schedule = [
                x for x in schedule_objs if x.Name == str(metabolic_schedule_id)
            ]
        else:
            metabolic_schedule = [
                s for s, i in zip(schedule_objs, index_first_value)
                if s.obj[i] > 0
            ]

        for s in metabolic_schedule:
            first = s.objls.index('Value_1')
            s.obj[first:first + len(schedule)] = schedule

    def change_thickness(self, material, thickness):
        target = [m for m in self.material if m.Name == material]
        if len(target) == 0:
            raise ValueError(f'대상 재료가 존재하지 않음: {material}')

        target[0].Thickness = thickness

    def change_window_u_value(self,
                              u_value,
                              obj='WindowMaterial:SimpleGlazingSystem'):
        windows = self._idf.idfobjects[obj]
        for w in windows:
            w.UFactor = u_value

    def change_year(self, u_value: float, materials: dict):
        """
        준공연도 변경

        Parameters
        ----------
        u_value : float
            window u-value
        materials : dict
            {material[str]: thickness[float]}
        """
        self.change_window_u_value(u_value=u_value)

        for material, thickness in materials.items():
            self.change_thickness(material=material, thickness=thickness)

        logger.debug('change year: u-value {}, materials {}', u_value,
                     materials)


def _read_bunches_helper(idf: IDF, key, name):
    result = idf.idfobjects[key]
    p = re.compile(name)

    return [x for x in result if p.match(x.Name)]


def read_bunches(idd: StrPath, idf: StrPath, bunch: dict):
    case = EnergyPlusCase(idd, idf)
    bunches = [_read_bunches_helper(case.idf, *x) for x in bunch.items()]

    return bunches


def save_as_eppy(idd: StrPath, idf: StrPath, output: Optional[StrPath] = None):
    idf = Path(idf)
    case = EnergyPlusCase(idd, idf)

    if not output:
        output = idf.parent.joinpath(f'{idf.stem}_eppy.idf')

    case.save(output)
