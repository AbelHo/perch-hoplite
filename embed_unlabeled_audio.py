# -*- coding: utf-8 -*-
"""
Script to embed unlabeled audio files using perch-hoplite.
This refactors the logic from v2_1_embed_unlabeled_audio.py into a single function for programmatic use.
"""
from etils import epath
import numpy as np
from perch_hoplite.agile import colab_utils, embed, source_info
from perch_hoplite.db import brutalism, interface


def embed_unlabeled_audio(
    dataset_name: str,
    dataset_base_path: str,
    dataset_fileglob: str = '*.flac',
    db_path: str = None,
    model_choice: str = 'perch_v2',
    use_file_sharding: bool = True,
    shard_length_in_seconds: float = 60.0,
    target_sample_rate_hz: int = -2,
    drop_existing_db: bool = False,
    verbose: bool = True
):
    """
    Embeds audio files into a perch-hoplite database.

    Args:
        dataset_name: Name for the dataset in the DB.
        dataset_base_path: Path to the folder containing audio files.
        dataset_fileglob: File pattern for audio files (e.g., '*.wav').
        db_path: Output DB directory. If None, uses dataset_base_path.
        model_choice: Model to use for embedding.
        use_file_sharding: Whether to split audio into shards.
        shard_length_in_seconds: Length of each shard in seconds.
        target_sample_rate_hz: Target sample rate for audio.
        drop_existing_db: If True, deletes existing DB before embedding.
        verbose: If True, prints progress and stats.

    Returns:
        db: The loaded HopliteDBInterface after embedding.
        stats: Dict with per-dataset embedding counts.
        search_example: Dict with example search results (embedding ids).
    """
    audio_glob = source_info.AudioSourceConfig(
        dataset_name=dataset_name,
        base_path=dataset_base_path,
        file_glob=dataset_fileglob,
        min_audio_len_s=1.0,
        target_sample_rate_hz=target_sample_rate_hz,
        shard_len_s=float(shard_length_in_seconds) if use_file_sharding else None,
    )

    configs = colab_utils.load_configs(
        source_info.AudioSources((audio_glob,)),
        db_path,
        model_config_key=model_choice,
        db_key='sqlite_usearch',
    )

    db = configs.db_config.load_db()
    num_embeddings = db.count_embeddings()

    if verbose:
        print('Initialized DB located at', configs.db_config.db_config.db_path)

    if num_embeddings > 0 and drop_existing_db:
        if verbose:
            print('Existing DB contains datasets:', db.get_dataset_names())
            print('num embeddings:', num_embeddings)
            print('Deleting previous db at:', configs.db_config.db_config.db_path)
        db_path_obj = epath.Path(configs.db_config.db_config.db_path)
        for fp in db_path_obj.glob('hoplite.sqlite*'):
            fp.unlink()
        index_fp = db_path_obj / 'usearch.index'
        if index_fp.exists():
            index_fp.unlink()
        db = configs.db_config.load_db()

    if verbose:
        print(f'Embedding dataset: {audio_glob.dataset_name}')

    worker = embed.EmbedWorker(
        audio_sources=configs.audio_sources_config,
        db=db,
        model_config=configs.model_config)

    worker.process_all(target_dataset_name=audio_glob.dataset_name)

    if verbose:
        print('\n\nEmbedding complete, total embeddings:', db.count_embeddings())

    stats = {}
    for dataset in db.get_dataset_names():
        count = db.get_embeddings_by_source(dataset, source_id=None).shape[0]
        stats[dataset] = count
        if verbose:
            print(f"\nDataset '{dataset}':")
            print('\tnum embeddings:', count)

    # Example embedding search
    # q = db.get_embedding(db.get_one_embedding_id())
    # results, scores = brutalism.brute_search(worker.db, query_embedding=q, search_list_size=128, score_fn=np.dot)
    # search_example = {
    #     'embedding_ids': [int(r.embedding_id) for r in results],
    #     'scores': scores.tolist() if hasattr(scores, 'tolist') else scores
    # }
    # if verbose:
    #     print('Example search embedding ids:', search_example['embedding_ids'])

    return db, stats#, search_example
