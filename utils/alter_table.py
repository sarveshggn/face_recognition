import sqlite3
# This script alters the 'photos' table in the 'photos.db' SQLite database

conn = sqlite3.connect('/home/sr/ov_fr/photos-test.db')
c = conn.cursor()

c.execute('''ALTER TABLE photos ADD COLUMN adaface_embedding_ov_int8''')
# conn.commit()

# c.execute('''UPDATE photos SET adaface_embedding = NULL''')

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Column added.")
