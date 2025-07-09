import os
import shutil
import json

with open("matches.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        img_path = data["img_path"]
        match_img_path = data['match_img_path']
        dest_path_split = img_path.split("/")
        src_path_split = match_img_path.split("/")
        photo_id1 = dest_path_split[-1]
        photo_id2 = src_path_split[-1]
        if (photo_id1[:4] != "IJBC" or photo_id2[:4] != "IJBC"):
            continue
        parts1 = photo_id1.split("_")
        parts2 = photo_id2.split("_")
        dest_fold_no = dest_path_split[-2]
        src_fold_no = src_path_split[-2]
        dest_folder = f"/home/sr/Aditya/Photos/{dest_fold_no}"
        source_folder = f"/home/sr/Aditya/Photos/{src_fold_no}"

        dest_no = parts1[1]
        # min_folder_no = 3585
        # max_folder_no = 3591
        # source_no = [str(i).zfill(6) for i in range(min_folder_no, max_folder_no + 1)]
        source_no = parts2[1]
        new_photo_no = 0
        dest_photo_dirs = sorted([f for f in os.listdir(dest_folder)])
        for img in dest_photo_dirs:
            if img.split("_")[1] == dest_no:
                new_photo_no = max(int(img.split("_")[2].split(".")[0]), new_photo_no)

        if new_photo_no == 0:
            continue
        new_photo_no = str(new_photo_no + 1).zfill(4)
        

        photo_dirs = sorted([f for f in os.listdir(source_folder)])

        for photo_dir in photo_dirs:
            old_path = os.path.join(source_folder, photo_dir)
            parts = photo_dir.split('_')
            if parts[1] != source_no:
                continue
            new_path = os.path.join(dest_folder, "IJBC"+"_"+dest_no+"_"+new_photo_no+".jpg")
            new_photo_no = int(new_photo_no) + 1
            new_photo_no = str(new_photo_no).zfill(4)
            shutil.move(old_path, new_path)
            print(old_path)
            
            
       
    