import os

folder = "/Users/odms/Documents/aditya_ws/Photos"
count1 = 0
count2 = 0
count3 = 0

photo_dirs = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])
for photo_dir in photo_dirs:
    photo_folder = os.path.join(folder, photo_dir)
    images = sorted([f for f in os.listdir(photo_folder)])
    for image in images:
        parts = image.split('_')
        if len(parts) < 3:
            continue
        if parts[2] == '0001.jpg' or parts[2] == '0001.png':
            count1+=1
        elif parts[2] == '0002.jpg' or parts[2] == '0002.png':
            count2+=1
        if parts[0] == 'IND':
            count3+=1

print(count1)
print(count2)
print(count3)