from getting_tess_df_in_batches import fetch_rrl_dataframe_in_batches
import numpy as np
from pathlib import Path
import random
from ramjet.data_interface.tess_data_interface import download_spoc_light_curves_for_tic_ids
cepheid_df = fetch_rrl_dataframe_in_batches(batch_size=2000)

tic_ids = cepheid_df.ID.unique()
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
    download_directory=Path('data_cepheid1/train_cepheid'),
    sectors=sectors,
    limit=2000)
download_spoc_light_curves_for_tic_ids(
    tic_ids=validation_tic_ids,
    download_directory=Path('data_cepheid1/validation_cepheid'),
    sectors=sectors,
    limit=2000)
download_spoc_light_curves_for_tic_ids(
    tic_ids=test_tic_ids,
    download_directory=Path('data_cepheid1/test_cepheid'),
    sectors=sectors,
    limit=2000)
#  _____________________________________________________________________