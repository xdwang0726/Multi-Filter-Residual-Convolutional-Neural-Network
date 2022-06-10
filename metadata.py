from collections import Counter

"""
metadata icd co-occurrence 
DRG 
{drg1: {icd1: count, icd2:count, ...}}
"""


def check_diagnosis_procedure(icd_list):
    new_list = []
    for icd in icd_list:
        if icd.find('.') == -1:
            continue
        else:
            if icd[3] == '.':
                new_list.append(icd)
    return new_list


def get_metadata_icd_occurrence(metadata, icds, diagnosis=True):
    meta_dict = {}
    meta_icd_counts = {}
    for i, data in enumerate(metadata):
        if isinstance(data, float) or isinstance(icds[i], float):
            continue
        else:
            data = data.split(';')
            for data_point in data:
                if data_point in meta_dict:
                    meta_dict[data_point]['counts'] = meta_dict[data_point]['counts'] + 1
                    # split icds codes into a list
                    icd = icds[i].split(';')
                    if diagnosis:
                        icd = check_diagnosis_procedure(icd)
                    meta_icd_counts[data_point].append(icd)
                else:
                    meta_dict[data_point] = dict.fromkeys(['counts', 'icd_counts'])
                    meta_dict[data_point]['counts'] = 1
                    icd = icds[i].split(';')
                    if diagnosis:
                        icd = check_diagnosis_procedure(icd)
                    meta_icd_counts[data_point] = [icd]

    for j, code in enumerate(list(meta_icd_counts.values())):
        flat_list = [item for sublist in code for item in sublist]
        occurrences = dict(Counter(flat_list))
        meta_name = list(meta_icd_counts.keys())[j]
        if meta_name in meta_dict:
            sorted_occurrence = dict(sorted(occurrences.items(), key=lambda item: item[1], reverse=True))
            meta_dict[meta_name]['icd_counts'] = sorted_occurrence
        else:
            print(meta_name)
    return meta_dict


def get_icd_from_metadata(metadata_dict, threshold):

    metadata_icd = {}
    for k, v in metadata_dict.items():
        num = v['counts']
        icd = []
        for i, (ids, counts) in enumerate(v['icd_counts'].items()):
            if list(v['icd_counts'].values())[i] / num >= threshold:
                icd.append(ids)
        metadata_icd[k] = icd

    return metadata_icd


def metadata_frequency(df, threshold=0.1):
    # df = pd.read_csv('%s/note.csv' % args.mimic_dir, dtype={'LABELS': str, 'DRG': str})
    icds = list(df['LABELS'])
    drgs = list(df['DRG'])
    prescription = list(df['PRESCRIPTIONS'])
    cpt = list(df['CPT'])

    drg_dict = get_metadata_icd_occurrence(drgs, icds, False) # get DRG co-occurrence with diagnosis codes
    drug_dict = get_metadata_icd_occurrence(prescription, icds, False) # get prescription co-occurrence with diagnosis codes
    cpt_dict = get_metadata_icd_occurrence(cpt, icds, False) # get prescription co-occurrence with diagnosis codes

    drg_icd = get_icd_from_metadata(drg_dict, threshold)
    drug_icd = get_icd_from_metadata(drug_dict, threshold)
    cpt_icd = get_icd_from_metadata(cpt_dict, threshold)

    return drg_icd, drug_icd, cpt_icd


def metadata_mask(metadata, metadata_dict):
    mask = []
    for i, data in enumerate(metadata):
        if isinstance(data, float):
            mask.append([])
        else:
            data = data.split(';')
            icd_from_metadata = []
            for data_point in data:
                icd_from_metadata += metadata_dict[data_point]
            mask.append(icd_from_metadata)
    return mask