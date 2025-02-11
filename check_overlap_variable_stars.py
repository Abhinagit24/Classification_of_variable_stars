from get_tic_ids_for_all_variables import get_tic_ids_from_fits

def check_for_overlap():
    """
    This function compares the tic ids of all short timescale variables with non-short timescale variables and print
    the results.
    """
    """'data_short_t/test_short_t', 'data_short_t/train_short_t', 'data_short_t/validation_short_t'"""
    tic_ids_for_delta = get_tic_ids_from_fits(['data_deltascuti/test_delta', 'data_deltascuti/train_delta', 'data_deltascuti/validation_delta'])
    tic_ids_for_non_delta = get_tic_ids_from_fits(
        ['data_cepheids/test_cepheids', 'data_cepheids/train_cepheids', 'data_cepheids/validation_cepheids', 'data_eclipsing/test_eclipsing','data_eclipsing/train_eclipsing', 'data_eclipsing/validation_eclipsing', 'data_rot_m/test_rot_m', 'data_rot_m/train_rot_m', 'data_rot_m/validation_rot_m',
         'data_rrlyrae/test_rrl','data_rrlyrae/train_rrl', 'data_rrlyrae/validation_rrl'])
    overlapping_tic_ids = list(set(tic_ids_for_delta) & set(tic_ids_for_non_delta))
    print(len(overlapping_tic_ids))
    return overlapping_tic_ids


def overlapping_data_by_class():
    """
    This function finds the files that are overlapping between the short_t and each non_short_t variable class.
    """
    tic_ids_for_short_t = get_tic_ids_from_fits(['data_deltascuti/test_delta', 'data_deltascuti/train_delta', 'data_deltascuti/validation_delta'])
    class_list = ['data_cepheids/test_cepheids', 'data_cepheids/train_cepheids', 'data_cepheids/validation_cepheids', 'data_eclipsing/test_eclipsing','data_eclipsing/train_eclipsing', 'data_eclipsing/validation_eclipsing', 'data_rot_m/test_rot_m', 'data_rot_m/train_rot_m', 'data_rot_m/validation_rot_m',
         'data_rrlyrae/test_rrl','data_rrlyrae/train_rrl', 'data_rrlyrae/validation_rrl']
    overlap_files = {}
    for x in class_list:
        class_name = f"{x}"
        overlap_files[class_name] = list(set(tic_ids_for_short_t) & set(get_tic_ids_from_fits([x])))
    print(overlap_files)
    return overlap_files
if __name__ == "__main__":
    check_for_overlap()
