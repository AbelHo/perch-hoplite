from embed_unlabled_audio import *



##############################################################################################################################
#~ user inputs ~#
dataset_base_paths = ['/large',
                      '/medium',
                      '/small']
dataset_names = ['large',
                 'medium',
                 'small']
db_paths = ['/db_path']
use_file_shardings = [False]
######### Example inputs #########
# dataset_base_paths = ['/1',
#                       '/2',
#                       '/3',
#                       '/4'
#                       ]
# dataset_names = ['1', '2', '3', '4']
dataset_fileglobs = ['*.flac']
# db_paths = ['']
model_choices = ['perch_v2']  # @param {type:'string'}
# use_file_shardings = [True]
shard_length_in_secondss = [60.0]
target_sample_rate_hzs = [-2]
drop_existing_dbs = [False]
verboses = [True]

##############################################################################################################################
#~ automate the expansion of single-element lists to match the length of the largest list ~#
# Expand all single-element lists to match the length of the largest list
list_vars = [
    dataset_base_paths,
    dataset_names,
    dataset_fileglobs,
    db_paths,
    model_choices,
    use_file_shardings,
    shard_length_in_secondss,
    target_sample_rate_hzs,
    drop_existing_dbs,
    verboses
]
max_len = max(len(lst) for lst in list_vars)

def expand_list(lst, n):
    return lst if len(lst) == n else lst * n if len(lst) == 1 else lst

dataset_base_paths = expand_list(dataset_base_paths, max_len)
dataset_names = expand_list(dataset_names, max_len)
dataset_fileglobs = expand_list(dataset_fileglobs, max_len)
db_paths = expand_list(db_paths, max_len)
model_choices = expand_list(model_choices, max_len)
use_file_shardings = expand_list(use_file_shardings, max_len)
shard_length_in_secondss = expand_list(shard_length_in_secondss, max_len)
target_sample_rate_hzs = expand_list(target_sample_rate_hzs, max_len)
drop_existing_dbs = expand_list(drop_existing_dbs, max_len)
verboses = expand_list(verboses, max_len)
#############################################################################################################################


for (dataset_base_path, dataset_name, dataset_fileglob, db_path, model_choice, use_file_sharding, shard_length_in_seconds, target_sample_rate_hz, drop_existing_db, verbose) in zip(
    dataset_base_paths,
    dataset_names,
    dataset_fileglobs,
    db_paths,
    model_choices,
    use_file_shardings,
    shard_length_in_secondss,
    target_sample_rate_hzs,
    drop_existing_dbs,
    verboses
):
    print(f"----------------- Processing dataset: {dataset_name} at {dataset_base_path}")
    embed_unlabeled_audio(
        dataset_name=dataset_name,
        dataset_base_path=dataset_base_path,
        dataset_fileglob=dataset_fileglob,
        db_path=db_path,
        model_choice=model_choice,
        use_file_sharding=use_file_sharding,
        shard_length_in_seconds=shard_length_in_seconds,
        target_sample_rate_hz=target_sample_rate_hz,
        drop_existing_db=drop_existing_db,
        verbose=verbose
    )