from perch_hoplite.db import interface, sqlite_usearch_impl, db_loader
import pandas as pd
import argparse

def set_sqlite_journal_mode_delete(db_dir, max_retries=5, retry_wait=1.0):
    """Set SQLite journal mode to DELETE for the hoplite.sqlite in db_dir, with retry if locked."""
    import sqlite3
    import os
    db_path = os.path.join(db_dir, "hoplite.sqlite")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=DELETE;")
    conn.commit()
    conn.close()

def insert_label_from_csv(db_dir, csv_path,
    annotator="annotators_past",
    id_colname='filename',
    vocalization_cols=None,
    species_cols=None,
    verbose=False,
    ):

    """
    For each embedding in the db, match source_id with id_colname in csv.
    For each of the vocalization and species columns, if True, insert label.
    """
    if vocalization_cols is None:
        vocalization_cols = [
            'harmonic', 'upsweep', 'concave', 'click', 'downsweep', 'sine', 'constant', 'clicktrain', 'individualclick',
            'convex', 'burstpulse', 'whistle', 'buzz', 'bark', 'quack', 'biphonal', 'overlap', 'grunt', 'uncertain'
        ]
    if species_cols is None:
        species_cols = ["SC","NP","OB","BE","DD","SL"]

    # Load DB
    db_config = {"db_path": db_dir}
    db = db_loader.DBConfig(db_key="sqlite_usearch", db_config=db_config).load_db()

    # Load CSV
    df = pd.read_csv(csv_path)
    if id_colname not in df.columns:
        raise ValueError(f"CSV must have a '{id_colname}' column")

    # Build a lookup for id_colname -> row
    df_lookup = df.set_index(id_colname)

    # For each embedding, get source_id, match to id_colname
    for embedding_id in db.get_embedding_ids():
        source = db.get_embedding_source(embedding_id)
        filename = source.source_id
        if filename not in df_lookup.index:
            continue  # skip if not in csv
        row = df_lookup.loc[filename]
        # Insert vocalization labels
        for col in vocalization_cols:
            if col in row.index and bool(row[col]):
                label = interface.Label(
                    embedding_id=embedding_id,
                    label=f"vocalization: {col}",
                    type=interface.LabelType.POSITIVE,
                    provenance=annotator
                )
                db.insert_label(label, skip_duplicates=True)
        # Insert species labels
        for col in species_cols:
            if col in row.index and bool(row[col]):
                label = interface.Label(
                    embedding_id=embedding_id,
                    label=f"species: {col}",
                    type=interface.LabelType.POSITIVE,
                    provenance=annotator
                )
                db.insert_label(label, skip_duplicates=True)


    db.commit()
    db.db.close()

    if verbose:
        print(f"Labels inserted into DB at {db_dir} from CSV {csv_path}")
        counts = db.get_class_counts()
        print("Label counts:")
        for label, count in counts.items():
            print(f"  {label}: {count}")

# Test the function
if __name__ == "__main__":
    db_dir = ""
    csv_path = ""
    parser = argparse.ArgumentParser(description="Insert labels from CSV into the database.")
    parser.add_argument("db_dir", type=str, nargs="?", default=db_dir, help="Path to the database directory (default: current directory)")
    parser.add_argument("csv_path", type=str, nargs="?", default=csv_path, help="Path to the CSV file (default: current directory)")
    args = parser.parse_args()

    insert_label_from_csv(args.db_dir, args.csv_path)
