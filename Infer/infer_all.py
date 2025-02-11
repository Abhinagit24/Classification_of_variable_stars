from functools import partial
from pathlib import Path

import numpy as np
import torch
import csv
from qusi.internal.light_curve_collection import LightCurveCollection
from qusi.internal.light_curve_dataset import default_light_curve_post_injection_transform

from qusi.session import get_device, infer_session
from qusi.internal.finite_standard_light_curve_observation_dataset import FiniteStandardLightCurveObservationDataset
from ramjet.photometric_database.tess_two_minute_cadence_light_curve import TessMissionLightCurve

from Hadryss_new import HadryssNew, HadryssMultiClassProbabilityEndModuleNew


def get_infer_paths():
    directory_1 = Path('/media/abhina/My Passport/Astroproject/data_negatives/test_negative')
    directory_2 = Path('/media/abhina/My Passport/Astroproject/data_negatives/train_negative')
    directory_3 = Path('/media/abhina/My Passport/Astroproject/data_negatives/validation_negative')


    paths = list(directory_1.glob('*.fits')) + \
            list(directory_2.glob('*.fits')) + \
            list(directory_3.glob('*.fits'))
    return paths


def load_times_and_fluxes_from_path(path: Path) -> (np.ndarray, np.ndarray):
    light_curve = TessMissionLightCurve.from_path(path)
    return light_curve.times, light_curve.fluxes


def infer():
    infer_light_curve_collection = LightCurveCollection.new(
        get_paths_function=get_infer_paths,
        load_times_and_fluxes_from_path_function=load_times_and_fluxes_from_path)

    test_light_curve_dataset = FiniteStandardLightCurveObservationDataset.new(
        light_curve_collections=[infer_light_curve_collection],
        post_injection_transform=partial(default_light_curve_post_injection_transform, length=3500, randomize=False))

    model = HadryssNew.new(end_module=HadryssMultiClassProbabilityEndModuleNew(number_of_classes=6))
    device = get_device()
    model.load_state_dict(
        torch.load('/home/abhina/Astroproject/sessions/usual-frog-99_latest_model.pt', map_location=device))
    confidences = infer_session(infer_datasets=[test_light_curve_dataset], model=model,
                                batch_size=100, device=device)[0]
    paths = list(get_infer_paths())
    paths_with_confidences = zip(paths, confidences)
    sorted_paths_with_confidences = sorted(
        paths_with_confidences, key=lambda path_with_confidence: path_with_confidence, reverse=True)
    print(sorted_paths_with_confidences)

    with open('infer_result_99_whole.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['path', 'confidence'])
        writer.writerows(sorted_paths_with_confidences)
    return sorted_paths_with_confidences


if __name__ == '__main__':
    infer()
