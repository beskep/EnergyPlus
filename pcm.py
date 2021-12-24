from datetime import datetime
from itertools import product
import os
import re

from eppy.bunch_subclass import EpBunch
import pandas as pd
from tqdm import tqdm

from energy_plus_case import EnergyPlusCase

temperature_pattern = re.compile('temperature_[0-9]+')


class PCMCase(EnergyPlusCase):

    def __init__(self,
                 idd_path,
                 idf_path,
                 epw_path=None,
                 output_template=None,
                 pcm_name=None):
        super(PCMCase, self).__init__(idd_path, idf_path, epw_path,
                                      output_template)

        mp_obj, mp_name = self._get_obj_and_name('MaterialProperty:PhaseChange')
        if not mp_name:
            raise ValueError('idf에 PCM이 존재하지 않습니다.')
        if pcm_name is None:
            if len(mp_name) == 1:
                self._pcm_name = mp_name[0]
            else:
                raise ValueError('idf에 PCM이 둘 이상 존재합니다. 대상 PCM 이름을 입력해주세요.')
        else:
            if pcm_name not in mp_name:
                raise ValueError('PCM 이름을 잘못 입력했습니다.')
            self._pcm_name = pcm_name

        self._pcm: EpBunch = mp_obj.list1[mp_name.index(self._pcm_name)]
        self._pcm_temperature_index = [
            i for i, x in enumerate(self._pcm.objls)
            if temperature_pattern.match(x.lower()) and i < len(self._pcm.obj)
        ]

        materials, names = self._get_obj_and_name('Material')
        self._pcm_material: EpBunch = materials.list1[names.index(
            self._pcm_name)]

    def change_pcm_temperature(self, temperature_change):
        for idx in self._pcm_temperature_index:
            self._pcm.obj[idx] += temperature_change

    def change_pcm_thickness(self, thickness):
        self._pcm_material.Thickness = thickness


class PCMRunner:

    def __init__(self,
                 idd_path,
                 idf_path,
                 material_path,
                 window_path,
                 pcm_name=None,
                 output_template=None,
                 remove_list=None,
                 epw_dir='./input/pcm_weather'):
        assert os.path.exists(idd_path)
        assert os.path.exists(idf_path)
        assert os.path.exists(material_path)
        assert os.path.exists(window_path)
        if output_template is not None:
            assert os.path.exists(output_template)

        self._idd_path = idd_path
        self._idf_path = idf_path
        self._material = pd.read_csv(material_path)
        self._window = pd.read_csv(window_path)
        self._pcm_name = pcm_name

        self._output_template = output_template
        self._remove_list = remove_list

        self._locations = list(pd.unique(self._material['location']))
        self._locations.sort()

        self._year = list(pd.unique(self._material['year']))
        self._year.sort()

        # material, window의 year, location 일치하는지 확인
        loc_window = list(pd.unique(self._window['location']))
        loc_window.sort()
        assert self._locations == loc_window
        year_window = list(pd.unique(self._window['year']))
        year_window.sort()
        assert self._year == year_window

        self._epw_dir = epw_dir
        for loc in self._locations:
            self.get_epw_path(loc)

    def get_epw_path(self, loc: str):
        path = os.path.join(self._epw_dir, loc + '.epw')
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return path

    def get_material_setting(self, year, location):
        mat = self._material.loc[(self._material['year'] == year) &
                                 (self._material['location'] == location), :]
        mat = mat.drop(columns=['year', 'location'])
        assert mat.shape[0] == 1
        mat_name = list(mat.columns)
        mat_thickness = list(mat.iloc[0, :])
        return mat_name, mat_thickness

    def get_window_u_value(self, year, location):
        window = self._window.loc[(self._window['year'] == year) &
                                  (self._window['location'] == location), :]
        assert window.shape[0] == 1
        return window['window'].iloc[0]

    def run(self,
            thickness=(0.01, 0.02, 0.03, 0.04),
            t_change=0.1,
            t_step=111,
            t_start=None,
            locations=None,
            year=None,
            run=True,
            save_idf=False,
            verbose=False):
        if locations is None:
            locations = self._locations.copy()
        if year is None:
            year = self._year.copy()

        output_path = os.path.normpath(os.path.abspath('./result/'))
        verbose_ep = 'v' if verbose else 'q'

        case = PCMCase(self._idd_path,
                       self._idf_path,
                       output_template=self._output_template,
                       pcm_name=self._pcm_name)
        if self._output_template is not None:
            case.set_output()
        if self._remove_list is not None:
            for x in self._remove_list:
                case.remove_obj(x)

        for idx in tqdm(range(t_step)):
            case.change_pcm_temperature(t_change)

            for yr, loc, thc in product(year, locations, thickness):
                case.change_pcm_thickness(thc)

                mat_name, mat_thickness = self.get_material_setting(yr, loc)
                for mn, mt in zip(mat_name, mat_thickness):
                    case.change_thickness(mn, mt)

                case.change_window_u_value(self.get_window_u_value(yr, loc))

                if t_start is None:
                    case_name = 'year{}_loc{}_thc{}_dt{:04.1f}'.format(
                        yr, loc, thc, idx * t_change)
                else:
                    case_name = 'year{}_loc{}_thc{}_t{:04.1f}'.format(
                        yr, loc, thc, t_start + idx * t_change)

                if save_idf:
                    case.idf.saveas(
                        os.path.join(output_path, case_name + '.idf'))

                if run:
                    try:
                        case.idf.run(
                            weather=self.get_epw_path(loc),
                            output_directory=output_path,
                            output_prefix='eplus_{}_'.format(case_name),
                            readvars=True,
                            verbose=verbose_ep)
                    except Exception as e:
                        with open('./error.txt', 'a') as f:
                            f.write('{} {}: {}\n'.format(
                                datetime.now(), case_name, e))


if __name__ == '__main__':
    idf_path = os.path.normpath(
        os.path.abspath('./idf/pcm_01_simulation(10).idf'))
    idd_path = os.path.normpath(os.path.abspath('./idd/V8-3-0-Energy+.idd'))

    material_path = './input/material_pcm.csv'
    window_path = './input/window_pcm.csv'

    out_template_path = os.path.normpath(
        os.path.abspath('./idf/output_template.idf'))
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

    runner = PCMRunner(idd_path,
                       idf_path,
                       material_path,
                       window_path,
                       output_template=out_template_path,
                       remove_list=remove_list)
    runner.run(thickness=(0.01,),
               t_change=1.0,
               t_step=1,
               t_start=None,
               locations=None,
               year=None,
               run=True,
               save_idf=True,
               verbose=True)
