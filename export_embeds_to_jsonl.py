import sqlite3
import json
import numpy as np

def export_embeddings_to_jsonl(db_path, output_jsonl_path):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query to fetch image paths and adaface embedding file paths
    cursor.execute("SELECT path, adaface_embedding_ov FROM photos")
    records = cursor.fetchall()

    with open(output_jsonl_path, 'w') as jsonl_file:
        for path, embedding_path in records:
            if not embedding_path:
                continue  # Skip if embedding path is null or empty

            try:
                # Load embedding from binary file
                embedding = np.fromfile(embedding_path, dtype=np.float32).tolist()

                # Construct dictionary
                record = {
                    "path": path,
                    "feat": embedding
                }

                # Write as JSON line
                jsonl_file.write(json.dumps(record) + '\n')

            except Exception as e:
                print(f"Error processing {embedding_path}: {e}")

    conn.close()
    print(f"✅ Export completed: {output_jsonl_path}")

# Example usage
if __name__ == "__main__":
    export_embeddings_to_jsonl(
        db_path="photos-test.db",
        output_jsonl_path="final_embeddings_adaface_ov.jsonl"
    )
