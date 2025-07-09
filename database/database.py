import sqlite3
from main import embed_image, comp
import os
import json
import argparse
import cv2
import numpy as np
from openvino.runtime import Core

conn = sqlite3.connect('photos.db')
c = conn.cursor()

c.execute(''' create table if not exists photos (
          path text primary key,
          deepinsight_embedding text,
          arcface_embedding text,
          ghostface_embedding text,
          width integer,
          height integer,
          image_quality text,
          age integer,
          gender text,
          ethnicity text)''')
conn.commit()

# c.execute(''' drop table if exists photos''')

deepinsight_path = "/Users/odms/Documents/aditya_ws/Embeddings/Deepinsight"
folder_no = 1

def go_through():
    src_folder = "/Users/odms/Documents/aditya_ws/Photos"
    photo_dirs = sorted([d for d in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, d))])
    i = 0
    skip = 0
    remove = 0
    for photo_dir in photo_dirs:
        photo_folder = os.path.join(src_folder, photo_dir)
        images = sorted([f for f in os.listdir(photo_folder)])
        
        
        for image in images:
            
            parts = image.split('_')
            if len(parts) < 3:  
                continue
            
            img_path = os.path.join(photo_folder, image)
            c.execute('''select * from photos where path = ?''', (img_path,))
            
            if c.fetchone() is not None:
                
                skip += 1
                continue
                
            print(img_path)
            try:
                height, width = cv2.imread(img_path).shape[:2]
            except Exception as e:
                print(f"Error reading image {img_path}: {e}")
                remove+=1
                os.remove(img_path)
                continue
            
                
            args = argparse.Namespace(img1=img_path)
            feat = embed_image(args)
            if feat is None:
                continue
            i+=1
            
            emb_part = image.split('.')[0]
            deep_path = os.path.join(deepinsight_path, f"{emb_part}.bin")
            with open(deep_path, 'wb') as f:
                
                f.write(feat)
            
            
            
            c.execute('''INSERT INTO photos (path, deepinsight_embedding, width, height)
                       VALUES (?, ?, ?, ?)''',
                       (img_path, deep_path, width, height) 
                      )
            conn.commit()
            
            


            if (i%100 == 0):
                print(i)

    print(f"Skipped {skip} images that already exist in the database.")
    print(f"Removed {remove} images that could not be read.")
    print(f"Total processed images: {i}")
    
    print(f"Processed {i} images.")
    c.execute(''' select count(*) from photos''')
    count = c.fetchone()[0]
    print(f"Total images in database: {count}")
    return


def check():
    exist = 0
    c.execute('''SELECT * FROM photos''')
    rows = c.fetchall()
    for row in rows:
        emb_path = row[1]
        # if os.path.exists(emb_path):
        #     exist+=1
        #     continue
        # file_path = row[0]
        # args = argparse.Namespace(img1=file_path)
        # feat = embed_image(args)
        # if feat is None:
        #     print(f"Failed to embed image: {file_path}")
        #     continue
        # print(emb_path)
        # with open(emb_path, 'wb') as f:
        #     f.write(feat)
        parts = emb_path.split('/')
        if parts[-2] != 'Deepinsight' or parts[-3] != 'Embeddings':
            print(f"Invalid path: {emb_path}")
            continue
            
        
    print(exist)
        


def ghostface_make():
    core = Core()
    ghostface_path = "/Users/odms/Documents/aditya_ws/Embeddings/GhostFace"
# model = core.read_model("ghostFace-cosFace.xml")
    model = core.read_model("ghostface_embedding_new_ep16.xml")
    compiled_model = core.compile_model(model, "CPU")
    input_name = compiled_model.input(0).any_name
    output_name = compiled_model.output(0)

# --- Load and preprocess face image ---
    def preprocess(image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (112, 112))  # Resize as required
        img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
        img = np.expand_dims(img, axis=0)  # Add batch dim
        return img  # Shape: (1, 112, 112, 3)
    src_folder = "/Users/odms/Documents/aditya_ws/Photos"
    photo_dirs = sorted([d for d in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, d))])
    i = 0
    skip = 0
    remove = 0
    for photo_dir in photo_dirs:
        photo_folder = os.path.join(src_folder, photo_dir)
        images = sorted([f for f in os.listdir(photo_folder)])
        
        
        for image in images:
            
            parts = image.split('_')
            if len(parts) < 3:  
                continue
            
            img_path = os.path.join(photo_folder, image)
            
                
            
            try:
                height, width = cv2.imread(img_path).shape[:2]
            except Exception as e:
                print(f"Error reading image {img_path}: {e}")
                remove+=1
                os.remove(img_path)
                continue
            
            
            img = preprocess(img_path)
            embedding = compiled_model({input_name: img})[output_name]
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            i+=1
            
            emb_part = image.split('.')[0]
            ghost_path = os.path.join(ghostface_path, f"{emb_part}.bin")
            with open(ghost_path, 'wb') as f:
                
                f.write(embedding)
            
            
            
            c.execute(''' update photos set ghostface_embedding = ? where path = ?''',
                       (ghost_path, img_path) 
                      )
            conn.commit()
            


            if (i%100 == 0):
                print(i)

    print(f"Skipped {skip} images that already exist in the database.")
    print(f"Removed {remove} images that could not be read.")
    print(f"Total processed images: {i}")
    
    print(f"Processed {i} images.")
    c.execute(''' select count(*) from photos''')
    count = c.fetchone()[0]
    print(f"Total images in database: {count}")
    return

# go_through()
# check()
# c.execute("SELECT * FROM photos ORDER BY path DESC LIMIT 1")
# last_row = c.fetchone()
# print(last_row)

ghostface_make()
c.execute('''SELECT COUNT(*) FROM photos''')
print(c.fetchone()[0])
# c.execute('''select * from photos''')
# rows = c.fetchall()
# for row in rows:
#     print(row)
#     break


# c.execute('''
#     SELECT deepinsight_embedding, COUNT(*) 
#     FROM photos 
#     GROUP BY deepinsight_embedding 
#     HAVING COUNT(*) > 1
# ''')
# duplicates = c.fetchall()
# for dup in duplicates:
#     print(f"Duplicate found:  embedding {dup[0]} appears {dup[1]} times.")

