from itertools import product
from pathlib import Path
import re

from eppy.bunch_subclass import EpBunch
from loguru import logger
import pandas as pd

from ep.ep import EnergyPlusCase
from ep.utils import track

temperature_pattern = re.compile('temperature_[0-9]+')


class PCMCase(EnergyPlusCase):

    def __init__(self,
                 idd_path,
                 idf_path,
                 epw_path=None,
                 output_template=None,
                 pcm_name=None):
        super().__init__(idd_path, idf_path, epw_path, output_template)

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

    def set_pcm_temperature(self, temperature_change):
        for idx in self._pcm_temperature_index:
            self._pcm.obj[idx] += temperature_change

    def set_pcm_thickness(self, thickness):
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
        self._idd_path = Path(idd_path)
        self._idf_path = Path(idf_path)
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

        self._epw_dir = Path(epw_dir).resolve()
        for loc in self._locations:
            self.get_epw_path(loc)

    def get_epw_path(self, loc: str):
        path = self._epw_dir.joinpath(f'{loc}.epw')
        path.stat()

        return path

    def get_material_setting(self, year, location):
        # pylint: disable=unsubscriptable-object
        mat = self._material.loc[(self._material['year'] == year) &
                                 (self._material['location'] == location), :]
        mat = mat.drop(columns=['year', 'location'])
        assert mat.shape[0] == 1
        mat_name = list(mat.columns)
        mat_thickness = list(mat.iloc[0, :])
        return mat_name, mat_thickness

    def get_window_u_value(self, year, location):
        # pylint: disable=unsubscriptable-object
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

        output_path = Path('./result/')
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

        for idx in track(range(t_step)):
            case.set_pcm_temperature(t_change)

            for yr, loc, thc in product(year, locations, thickness):
                case.set_pcm_thickness(thc)

                mat_name, mat_thickness = self.get_material_setting(yr, loc)
                for mn, mt in zip(mat_name, mat_thickness):
                    case.set_material_thickness(mn, mt)

                case.set_window_u_value(self.get_window_u_value(yr, loc))

                if t_start is None:
                    ts = f'dt{idx*t_change:04.1f}'
                else:
                    ts = f't{t_start + idx*t_change:04.1f}'
                case_name = f'year{yr}_loc{loc}_thc{thc}_{ts}'

                if save_idf:
                    case.idf.saveas(
                        output_path.joinpath(f'{case_name}.idf').as_posix())

                if run:
                    try:
                        case.idf.run(weather=self.get_epw_path(loc),
                                     output_directory=output_path,
                                     output_prefix=f'eplus_{case_name}_',
                                     readvars=True,
                                     verbose=verbose_ep)
                    except Exception as e:  # pylint: disable=broad-except
                        logger.exception(e)
