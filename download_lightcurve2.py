'''
This python file takes the csv file with the source_ids of GAIA objects and get their corresponding TIC ids and later
use those TIC ids to get the TESS row for each object. This data is returned as a dataframe. Further, the TIC ids in the
dataframe are used to download the lightcurves
'''
import random
from pathlib import Path

import numpy as np
import pandas as pd
from ramjet.data_interface.tess_data_interface import download_spoc_light_curves_for_tic_ids

#from getting_tess_df_in_batches import fetch_rrl_dataframe_in_batches
#df = fetch_rrl_dataframe_in_batches(batch_size=1500)
random.seed(42)
df = pd.read_csv(input('enter the csv file name'))
tic_ids = df.iloc[:, 0].unique()
random.shuffle(tic_ids)
# splitting the tic_ids as train-validation-test
tic_ids_splits = np.split(np.array(tic_ids), [int(len(tic_ids) * 0.8), int(len(tic_ids) * 0.9)])
train_tic_ids = tic_ids_splits[0].tolist()
validation_tic_ids = tic_ids_splits[1].tolist()
test_tic_ids = tic_ids_splits[2].tolist()
sectors = list(range(27, 56))
print('Retrieving metadata...')

print(f'Downloading light curve for {tic_ids}')
download_spoc_light_curves_for_tic_ids(
    tic_ids=train_tic_ids,
    download_directory=Path('/media/abhina/Astroproject/mrt_cepheid/train_cepheid'),
    sectors=sectors,
    limit=2000)
download_spoc_light_curves_for_tic_ids(
    tic_ids=validation_tic_ids,
    download_directory=Path('/media/abhina/Astroproject/mrt_cepheid/validation_cepheid'),
    sectors=sectors,
    limit=2000)
download_spoc_light_curves_for_tic_ids(
    tic_ids=test_tic_ids,
    download_directory=Path('/media/abhina/Astroproject/mrt_cepheid/test_cepheid'),
    sectors=sectors,
    limit=2000)
#  _____________________________________________________________________
# '''
