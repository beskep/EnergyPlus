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
        if self._template:
            self.set_output()

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

    def _get_objs(self, obj) -> Idf_MSequence:
        objs: Idf_MSequence = self._idf.idfobjects[obj]
        if not objs.list1:
            raise ValueError(f'"{obj}" not found')

        return objs

    def _get_obj_and_name(self, obj):
        # TODO remove
        objs = self._get_objs(obj)
        names: tuple[str, ...] = tuple(x.Name for x in objs)

        return objs, names

    def set_output(self, variable='Output:Variable'):
        self._idf.idfobjects[variable] = self._template.idfobjects[variable]

    def remove_obj(self, variable):
        obj = self._idf.idfobjects[variable]
        if len(obj):
            self._idf.removeidfobject(obj[0])

    def set_north_axis(self, north_axis):
        building = self._idf.idfobjects['BUILDING'][0]
        building.North_Axis = north_axis

    def set_infiltration(self, infiltration_rate: float):
        """
        Parameters
        ----------
        infiltration_rate : float
            [1/h]
        """
        ir = infiltration_rate / 3600.0  # [1/sec]

        for zi in self._get_objs('ZoneInfiltration:DesignFlowRate'):
            volume = self.zone(zi.Zone_or_ZoneList_Name).Volume
            zi.Design_Flow_Rate = volume * ir

    def set_occupancy(self, density: float):
        """
        Parameters
        ----------
        density : float
            [people/m^2]
        """
        for people in self._get_objs('PEOPLE'):
            area = self.zone(people.Zone_or_ZoneList_Name).Floor_Area
            people.Number_of_People = density * area

    def set_equipment_power(self, power: float):
        """
        Parameters
        ----------
        power : float
            [W/m^2]
        """
        for equipment in self._get_objs('ElectricEquipment'):
            area = self.zone(equipment.Zone_or_ZoneList_Name).Floor_Area
            equipment.Design_Level = power * area

    def set_lighting_level(self, power: float):
        """
        Parameters
        ----------
        power : float
            [W/m^2]
        """
        for lights in self._get_objs('Lights'):
            area = self.zone(lights.Zone_or_ZoneList_Name).Floor_Area
            lights.Lighting_Level = area * power

    def set_material_thickness(self, material, thickness):
        target = [m for m in self.material if m.Name == material]
        if len(target) == 0:
            raise ValueError(f'대상 재료가 존재하지 않음: {material}')

        target[0].Thickness = thickness

    def set_simple_glazing_system(self,
                                  u_value: Optional[float] = None,
                                  shgc: Optional[float] = None,
                                  transmittance: Optional[float] = None,
                                  name: Optional[str] = None):
        """
        Set `WindowMaterial:SimpleGlazingSystem`

        Parameters
        ----------
        u_value : Optional[float], optional
            U-value [W/m2/K]
        shgc : Optional[float], optional
            Solar Heat Gain Coefficient
        transmittance : Optional[float], optional
            Visible Transmittance
        name : Optional[str], optional
            Object Name. `None`이면 모든 `WindowMaterial:SimpleGlazingSystem`를 변경.
        """
        windows = self._get_objs('WindowMaterial:SimpleGlazingSystem')

        for window in windows:
            if name and window.Name != name:
                continue

            if u_value is not None:
                window.UFactor = u_value

            if shgc is not None:
                window.Solar_Heat_Gain_Coefficient = shgc

            if transmittance is not None:
                window.Visible_Transmittance = transmittance

    def set_year(self, u_value: float, materials: dict):
        """
        준공연도 변경

        Parameters
        ----------
        u_value : float
            window u-value
        materials : dict
            {material[str]: thickness[float]}
        """
        self.set_simple_glazing_system(u_value=u_value)

        for material, thickness in materials.items():
            self.set_material_thickness(material=material, thickness=thickness)


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
