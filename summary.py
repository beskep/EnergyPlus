import os
import re

import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    pattern_path = re.compile('^eplus_y(.*)_loc(.*)_a(.*)_p(.*)'
                              '_s(.*)_b(.*)_led(.*)_w(.*)_i(.*)_out\.csv')

    pattern_bz = re.compile('\d.*ZONE\d+', re.IGNORECASE)
    pattern_zone = re.compile(':ZONE\d+', re.IGNORECASE)
    pattern_equipment = re.compile('ELECTRIC EQUIPMENT#')

    case_list = os.listdir('./result')
    case_list = [x for x in case_list if pattern_path.match(x)]

    results = []

    for case in tqdm(case_list):
        result = pd.read_csv(os.path.join('./result', case))
        # result = result.loc[result['Date/Time'] == 'December', :]

        result_melt = pd.melt(result, id_vars=['Date/Time'], var_name='bzv')
        result_melt['bzv'] = [
            pattern_equipment.sub('', x) for x in result_melt['bzv']
        ]

        mask = [
            bool(pattern_bz.findall(x)) or x == 'Date/Time'
            for x in result_melt['bzv']
        ]
        result_melt: pd.DataFrame = result_melt.loc[mask, :]

        result_melt['bz'] = [
            pattern_bz.findall(x)[0].strip() for x in result_melt['bzv']
        ]
        result_melt['building'] = [
            pattern_zone.sub('', x).strip().upper() for x in result_melt['bz']
        ]
        result_melt['variable'] = [
            pattern_bz.sub('', x).strip() for x in result_melt['bzv']
        ]

        result_sum = result_melt.groupby(['building',
                                          'variable']).sum().reset_index()
        result_sum['case'] = case
        result_sum = result_sum.loc[:,
                                    ['case', 'building', 'variable', 'value']]

        results.append(result_sum)

    summary: pd.DataFrame = pd.concat(results)
    summary = summary.pivot_table(values='value',
                                  index=['case', 'building'],
                                  columns='variable')
    summary = summary.reset_index()

    case_variables = [
        'year', 'loc', 'azimuth', 'people', 'schedule', 'blind', 'LED',
        'window', 'insulation'
    ]
    cols = summary.columns

    for idx, var in enumerate(case_variables):
        summary[var] = [
            pattern_path.sub('\\' + str(idx + 1), x) for x in summary['case']
        ]

    summary = summary[case_variables + list(cols)]

    new_cols = [
        'Total Heating Energy', 'Total Cooling Energy',
        'InteriorEquipment:Electricity',
        'GeneralLights:InteriorLights:Electricity'
    ]
    for col in new_cols:
        search = [x for x in summary.columns if re.search(col, x)]
        summary[col + ' [kWh]'] = summary[search[0]] * 2.7777778 / (10**7)

    summary.replace('Na', pd.NaT, inplace=True)
    summary.to_csv('./result/_summary.csv', index=False)
