import os
import re

from eppy.modeleditor import IDF


class EnergyPlusCase:

    def __init__(self, idd_path, idf_path, epw_path=None, output_template=None):
        if IDF.getiddname() is None:
            IDF.setiddname(os.path.abspath(idd_path))
        if epw_path is None:
            epw_path = ''
        self._idf = IDF(idfname=os.path.abspath(idf_path),
                        epw=os.path.abspath(epw_path))

        self._zone = None
        self._zone_name = None
        self._material = None

        self._out_template = None if output_template is None else IDF(
            output_template)

    @property
    def idf(self):
        return self._idf

    def save(self, path):
        if not path:
            raise ValueError

        self._idf.saveas(path)

    def _get_obj_and_name(self, obj):
        obj_list = self._idf.idfobjects[obj]
        name_list = tuple(x.Name for x in obj_list)
        return obj_list, name_list

    def _get_zone(self):
        if self._zone is None:
            self._zone = self._idf.idfobjects['ZONE']
            self._zone_name = tuple(z.Name for z in self._zone)

    def set_output(self, variable='Output:Variable'):
        self._idf.idfobjects[variable] = self._out_template.idfobjects[variable]

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
        self._get_zone()
        # self._get_zone_infiltration()

        pattern = re.compile('\sInfiltration$')
        zi, zi_name = self._get_obj_and_name('ZoneInfiltration:DesignFlowRate')
        zi_name = tuple(pattern.sub('', x) for x in zi_name)

        zi_index = [self._zone_name.index(z) for z in zi_name]
        zone_volume = tuple(z.Volume for z in self._zone)

        for idx, zone in zip(zi_index, zi):
            zone.Design_Flow_Rate = zone_volume[idx] * infiltration_rate / 3600.0

    def change_people_density(self, density: float):
        """
        Parameters
        ----------
        density : float
            [people/m^2]
        """
        self._get_zone()
        # self._get_people()

        pattern = re.compile('^People\s')
        people, people_name = self._get_obj_and_name('PEOPLE')
        people_name = tuple(pattern.sub('', x) for x in people_name)

        people_index = [self._zone_name.index(n) for n in people_name]

        zone_area = tuple(z.Floor_Area for z in self._zone)
        for idx, people in zip(people_index, people):
            people.Number_of_People = density * zone_area[idx]

    def change_equipment_power(self, power: float):
        """
        Parameters
        ----------
        power : float
            [W/m^2]
        """
        self._get_zone()

        pattern = re.compile('\sEquipment\s\d+.*$')
        equipment, equipment_name = self._get_obj_and_name('ElectricEquipment')
        equipment_name = tuple(pattern.sub('', x) for x in equipment_name)

        equipment_index = [self._zone_name.index(n) for n in equipment_name]
        zone_area = tuple(z.Floor_Area for z in self._zone)

        for idx, eq in zip(equipment_index, equipment):
            eq.Design_Level = power * zone_area[idx]

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
        if self._material is None:
            self._material = self._idf.idfobjects['Material']

        target = [m for m in self._material if m.Name == material]

        if len(target) != 0:
            target[0].Thickness = thickness

    def change_window_u_value(self,
                              u_value,
                              obj='WindowMaterial:SimpleGlazingSystem'):
        windows = self._idf.idfobjects[obj]
        for w in windows:
            w.UFactor = u_value
        return


def _read_bunches_helper(idf: IDF, key, name):
    result = idf.idfobjects[key]
    p = re.compile(name)
    return [x for x in result if p.match(x.Name)]


def read_bunches(idd_path, idf_path, bunch: dict):
    case = EnergyPlusCase(idd_path, idf_path)
    result = [_read_bunches_helper(case.idf, *x) for x in bunch.items()]
    return result


def save_as_eppy(idd_path, idf_path):
    case = EnergyPlusCase(idd_path, idf_path)
    case.save(os.path.splitext(idf_path)[0] + '_eppy.idf')


if __name__ == '__main__':
    idf_path = os.path.abspath('./idf/59_NOT.idf')
    idd_path = os.path.abspath('./idd/V8-3-0-Energy+.idd')
    epw_path = os.path.abspath('./input/Hybrid Seoul-hourEPW.epw')
    out_template_path = os.path.abspath('./idf/output_template.idf')

    idfs = [
        './idf/01_pcm_Basic_simulation.idf',
        './idf/01_pcm_simulation(10-25.1).idf',
        './idf/01_pcm_simulation(20_25).idf',
        './idf/01_pcm_simulation(20-25.1).idf'
    ]

    for idf in idfs:
        save_as_eppy(idd_path, idf)
