#!/usr/bin/env python

import os
import os.path as osp
import argparse
import cv2
import numpy as np
import onnxruntime
# from scrfd import SCRFD
# from arcface_onnx import ArcFaceONNX
# from adaface_onnx import AdaFaceONNX
from adaface_openvino import AdaFaceOpenVINO
import json
import heapq
import faiss
import shutil
import cv2
import numpy as np
import faiss
import json
import time
# from scrfd_openvino_single_detect import SCRFD
# from scrfd_openvino_custom import SCRFD
from scrfd_openvino_sd_blur_detect import SCRFD

onnxruntime.set_default_logger_severity(3)

assets_dir = osp.expanduser('buffalo_l')

# detector = SCRFD(os.path.join(assets_dir, 'det_10g.onnx'))
detector = SCRFD(os.path.join(assets_dir, './FD/model.xml'))
# detector.prepare(0, blur_threshold=100)
# model_path = os.path.join(assets_dir, 'w600k_r50.onnx')
# rec = ArcFaceONNX(model_path)
adaface_model_path = os.path.join(assets_dir, 'adaface_ir_50_model_fp16.xml')
# adaface_model_path = os.path.join(assets_dir, 'adaface_ir_50_model.onnx')
# rec = AdaFaceONNX(adaface_model_path)
rec = AdaFaceOpenVINO(adaface_model_path)
rec.prepare(0)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('img1', type=str)
    parser.add_argument('img2', type=str)
    return parser.parse_args()


def func(args):
    image1 = cv2.imread(args.img1)
    image2 = cv2.imread(args.img2)
    bboxes1, kpss1 = detector.autodetect(image1, max_num=1)
    if bboxes1.shape[0]==0:
        return -1.0, "Face not found in Image-1"
    bboxes2, kpss2 = detector.autodetect(image2, max_num=1)
    if bboxes2.shape[0]==0:
        return -1.0, "Face not found in Image-2"
    kps1 = kpss1[0]
    kps2 = kpss2[0]
    feat1 = rec.get(image1, kps1)
    feat2 = rec.get(image2, kps2)
    sim = rec.compute_sim(feat1, feat2)
    if sim<0.2:
        conclu = 'They are NOT the same person'
    elif sim>=0.2 and sim<0.28:
        conclu = 'They are LIKELY TO be the same person'
    else:
        conclu = 'They ARE the same person'
    return sim, conclu

def embed_image(args):
    image = cv2.imread(args.img1)
    bboxes, kpss = detector.autodetect(image, max_num=1)
    if bboxes.shape[0]==0:
        return None
    kps = kpss[0]
    feat = rec.get(image, kps)
    return feat

def go_through():
    src_folder = "/Users/odms/Documents/aditya_ws/Photos"
    photo_dirs = sorted([d for d in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, d))])
    i = 0
    
    for photo_dir in photo_dirs[636:]:
        photo_folder = os.path.join(src_folder, photo_dir)
        images = sorted([f for f in os.listdir(photo_folder)])
        output = []
        
        for image in images:
            
            parts = image.split('_')
            if len(parts) < 3:
                continue
            # temp = parts[2].split('.')
            # if parts[2] == '0001.jpg':
            #     continue
            # if int(temp[0]) > 10:
            #     continue

            img_path = os.path.join(photo_folder, image)
            
            
            args = argparse.Namespace(img1=img_path)
            feat = embed_image(args)
            
        
            if feat is None:
                with open('failed_images.txt', 'a') as f:
                    f.write(img_path + '\n')
                continue
            i+=1
            output.append({'path': img_path, 'feat': feat.tolist()})
            if (i%1000 == 0):
                print(i)

        with open('final_embeddings.jsonl', 'a') as f:
            for item in output:
                f.write(json.dumps(item) + '\n')
    
            
    return i


def compare():
    src_folder = "/Users/odms/Documents/aditya_ws/Photos"
    photo_dirs = sorted([d for d in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, d))])
    i = 0
    fol_no = 0

    for photo_dir in photo_dirs[1:2]:
        photo_folder = os.path.join(src_folder, photo_dir)
        images = sorted([f for f in os.listdir(photo_folder)])
        # output = []
        fol_no+=1
        for image in images[558:]:
            
            parts = image.split('_')
            temp = parts[2].split('.')
            if parts[2] != '0001.jpg':
                continue

            img_path = os.path.join(photo_folder, image)
            args = argparse.Namespace(img1=img_path)
            feat = embed_image(args)
            if feat is None:
                print(img_path)
                continue

            top_photos = []
            counter = 0
            with open('embeddings.jsonl', 'r') as f:
                for line in f:
                    data = json.loads(line)
                    split_img = data['path'].split('_')
                    img_no = split_img[2].split('.')[0]
                    if (int(img_no )> 5) or parts[1] == split_img[1]:
                        continue
                    
                    vec = np.array(data['feat'])
                    sim = rec.compute_sim(feat, vec)
                    if sim <0.2:
                        continue
                    
                    
                    if len(top_photos) < 5:
                        heapq.heappush(top_photos, (sim, counter, {'img_path' : img_path, 'match_img_path' : data['path'], 'sim' : sim}))
                    else:
                        heapq.heappushpop(top_photos, (sim, counter, {'img_path' : img_path, 'match_img_path' : data['path'], 'sim' : sim}))
                    counter+=1
                    # top_photos.append(sim, {'img_path' : img_path, 'match_img_path' : data['path'], 'sim' : sim})
            
                    

            
            i+=1
            if len(top_photos) == 0:
                continue

            with open('matches.jsonl', 'a') as f:
                for item in top_photos:
                    f.write(json.dumps(item[2]) + '\n')
            if (i%10 == 0):
                print(i)
            # output.append({'path': img_path, 'feat': feat.tolist()})

        # with open('embeddings.jsonl', 'a') as f:
        #     for item in output:
        #         f.write(json.dumps(item) + '\n')
        # print(fol_no)
            
    return i



def compare2():
    src_folder = "/home/sr/Aditya/Photos"
    photo_dirs = sorted([d for d in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, d))])
    i = 0
    fol_no = 0

    with open('embeddings.jsonl', 'r') as f:
        for photo_dir in photo_dirs[1:2]:
            photo_folder = os.path.join(src_folder, photo_dir)
            images = sorted([f for f in os.listdir(photo_folder)])
        # output = []
            fol_no+=1
            for image in images[555:]:
                parts = image.split('_')
                if parts[2] != '0001.jpg':
                    continue

                img_path = os.path.join(photo_folder, image)
                args = argparse.Namespace(img1=img_path)
                feat = embed_image(args)
                if feat is None:
                    print(img_path)
                    continue

                top_photos = []
                counter = 0
                f.seek(0)
                for line in f:
                    data = json.loads(line)
                    split_img = data['path'].split('_')
                    img_no = split_img[2].split('.')[0]
                    if (int(img_no )> 5) or parts[1] == split_img[1]:
                        continue
                            
                    vec = np.array(data['feat'])
                    sim = rec.compute_sim(feat, vec)
                    if sim <0.2:
                        continue
                            
                            
                    if len(top_photos) < 5:
                        heapq.heappush(top_photos, (sim, counter, {'img_path' : img_path, 'match_img_path' : data['path'], 'sim' : sim}))
                    else:
                        heapq.heappushpop(top_photos, (sim, counter, {'img_path' : img_path, 'match_img_path' : data['path'], 'sim' : sim}))
                    counter+=1
                            # top_photos.append(sim, {'img_path' : img_path, 'match_img_path' : data['path'], 'sim' : sim})
            
                    

            
                i+=1
                if len(top_photos) == 0:
                    continue

                with open('matches.jsonl', 'a') as f:
                    for item in top_photos:
                        f.write(json.dumps(item[2]) + '\n')
                if (i%10 == 0):
                    print(i)

def compare3(threshold):
    src_folder = "/Users/odms/Documents/aditya_ws/Photos"
    photo_dirs = sorted([d for d in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, d))])
    i = 0
    
    true_negative = 0
    false_negative = 0
    false_positive = 0
    true_positive = 0

    embeddings = []
    
    with open('final_embeddings.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            path = data['path']
            feats = np.array(data['feat'])
            img_data = path.split('/')[-1]
            parts = img_data.split('_')
            person_no = parts[1]
            img_no = parts[2].split('.')[0]
            if (int(img_no) == 1):
                continue
            embeddings.append({'img_path' : path, 'feat' : feats, 'person_no' : person_no, 'img_no' : img_no, 'img_path': path})
    
    

    for photo_dir in photo_dirs:
        photo_folder = os.path.join(src_folder, photo_dir)
        images = sorted([f for f in os.listdir(photo_folder)])
        
        
        for image in images:
            
            img_details = image.split('/')[-1]
            parts = img_details.split('_')
            if parts[2] != '0001.jpg' and parts[2] != '0001.png':
                continue

            img_path = os.path.join(photo_folder, image)
            args = argparse.Namespace(img1=img_path)
            feat = embed_image(args)
            if feat is None:
                print(img_path)
                continue

            top_photo = {"current_photo": img_path, "match_photo": None, "sim": float('-inf')}
            
            for emb in embeddings:
                if (emb['person_no'] == parts[1] and emb['img_no'] == parts[2].split('.')[0]):
                    continue
                sim = rec.compute_sim(feat, emb['feat'])
                
                if sim > top_photo['sim']:
                    top_photo['sim'] = sim
                    top_photo['match_photo'] = emb['img_path']
                    
                
            
            i+=1
            
            if (i%100 == 0):
                print(i)

            match_sim = top_photo['sim']
            det = top_photo['match_photo'].split('/')[-1]
            match_person_no = det.split('_')[1]
            
            if match_sim >=threshold and match_person_no == parts[1]:
                true_positive += 1
                with open('true_positive.jsonl', 'a') as f:
                    f.write(json.dumps(top_photo) + '\n')
            
            elif match_sim >= threshold:
                false_positive += 1
                
                with open('false_positive.jsonl', 'a') as f:
                    f.write(json.dumps(top_photo) + '\n')

            elif match_sim < threshold and match_person_no == parts[1]:
                false_negative += 1
                with open('false_negative.jsonl', 'a') as f:
                    f.write(json.dumps(top_photo) + '\n')

            else:
                true_negative += 1
                with open('true_negative.jsonl', 'a') as f:
                    f.write(json.dumps(top_photo) + '\n')
            
            
            
    
    print(f'True Positive: {true_positive}, False Positive: {false_positive}, False Negative: {false_negative}, True Negative: {true_negative}')


def compare4(threshold):
    src_folder = "/Users/odms/Documents/aditya_ws/Photos"
    photo_dirs = sorted([d for d in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, d))])
    
    
    true_negative = 0
    false_negative = 0
    false_positive = 0
    true_positive = 0

    
    img_paths = []
    person_ids = []
    all_feats = []
    img_nos = []
    
    with open('final_embeddings_adaface_r101.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            path = data['path']
            feats = np.array(data['feat'])
            img_data = path.split('/')[-1]
            parts = img_data.split('_')
            person_no = parts[1]
            img_no = parts[2].split('.')[0]
            if (int(img_no) == 1 or int(img_no) == 2):
                continue
            img_paths.append(path)
            all_feats.append(feats)
            person_ids.append(person_no)
            img_nos.append(img_no)
            # embeddings.append({'img_path' : path, 'feat' : feats, 'person_no' : person_no, 'img_no' : img_no, 'img_path': path})
    
    features_matrix = np.stack(all_feats).astype('float32')
    dim = features_matrix.shape[1]
    faiss.normalize_L2(features_matrix)
    index = faiss.IndexFlatIP(dim)
    index.add(features_matrix)
    


    for photo_dir in photo_dirs[636:]:
        photo_folder = os.path.join(src_folder, photo_dir)
        images = sorted([f for f in os.listdir(photo_folder)])
        
        
        for image in images:
            
            img_details = image.split('/')[-1]
            parts = img_details.split('_')
            if len(parts) < 3:
                continue
            val = parts[2].split('.')[0]
            if int(val) != 1 and int(val) != 2:
                continue

            img_path = os.path.join(photo_folder, image)
            args = argparse.Namespace(img1=img_path)
            feat = embed_image(args)
            if feat is None:
                
                continue

            
            
            # for emb in embeddings:
            #     if (emb['person_no'] == parts[1] and emb['img_no'] == parts[2].split('.')[0]):
            #         continue
            #     sim = rec.compute_sim(feat, emb['feat'])
                
            #     if sim > top_photo['sim']:
            #         top_photo['sim'] = sim
            #         top_photo['match_photo'] = emb['img_path']
                    
            feat = np.array(feat, dtype='float32').reshape(1, -1)
            faiss.normalize_L2(feat)
            D, I = index.search(feat, 1)
            sim = float(D[0][0])
            index_id = int(I[0][0])

            
                
            
            
            

            match_sim = sim
            det = img_paths[index_id].split('/')[-1]
            match_person_no = det.split('_')[1]
            top_photo = {"current_photo": img_path, "match_photo": img_paths[index_id], "sim": match_sim}
            
            if match_sim >=threshold and match_person_no == parts[1]:
                true_positive += 1
                with open(f'new_true_positive_{threshold}.jsonl', 'a') as f:
                    f.write(json.dumps(top_photo) + '\n')
            
            elif match_sim >= threshold:
                false_positive += 1
                
                with open(f'new_false_positive_{threshold}.jsonl', 'a') as f:
                    f.write(json.dumps(top_photo) + '\n')

            elif match_sim < threshold and match_person_no == parts[1]:
                false_negative += 1
                with open(f'new_false_negative_{threshold}.jsonl', 'a') as f:
                    f.write(json.dumps(top_photo) + '\n')

            else:
                true_negative += 1
                with open(f'new_true_negative_{threshold}.jsonl', 'a') as f:
                    f.write(json.dumps(top_photo) + '\n')
            
            
            
    
    print(f'True Positive: {true_positive}, False Positive: {false_positive}, False Negative: {false_negative}, True Negative: {true_negative}, Threshold: {threshold}')


def move():
    src_folder = "/Users/odms/Documents/aditya_ws/Photos"
    photo_dirs = sorted([d for d in os.listdir(src_folder) if os.path.isdir(os.path.join(src_folder, d))])
    


    img_paths = []
    person_ids = []
    all_feats = []
    img_nos = []
    
    with open('final_embeddings2.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            path = data['path']
            feats = np.array(data['feat'])
            img_data = path.split('/')[-1]
            parts = img_data.split('_')
            person_no = parts[1]
            img_no = parts[2].split('.')[0]
            if (int(img_no) == 1):
                continue
            img_paths.append(path)
            all_feats.append(feats)
            person_ids.append(person_no)
            img_nos.append(img_no)
    
    features_matrix = np.stack(all_feats).astype('float32')
    dim = features_matrix.shape[1]
    faiss.normalize_L2(features_matrix)
    index = faiss.IndexFlatIP(dim)
    index.add(features_matrix)
    

    for photo_dir in photo_dirs:
        photo_folder = os.path.join(src_folder, photo_dir)
        images = sorted([f for f in os.listdir(photo_folder)])
        
        
        for image in images:
            count = 1000
            img_details = image.split('/')[-1]
            parts = img_details.split('_')
            img_path = os.path.join(photo_folder, image)
            if not os.path.exists(img_path):
                continue
            
            if len(parts) < 3:
                continue
            if parts[2] != '0001.jpg' and parts[2] != '0001.png':
                continue

            
            args = argparse.Namespace(img1=img_path)
            feat = embed_image(args)
            if feat is None:
                print(img_path)
                continue

                    
            feat = np.array(feat, dtype='float32').reshape(1, -1)
            faiss.normalize_L2(feat)
            D, I = index.search(feat, 50)
            
            matches = []
            for sim, index_id in zip(D[0], I[0]):
                if sim < 0.7:
                    continue
                match_sim = sim
                det = img_paths[index_id].split('/')[-1]
                match_person_no = det.split('_')[1]
                if match_person_no == parts[1]:
                    continue
                top_photo = {"match_photo": img_paths[index_id], "sim": float(match_sim)}
                matches.append(top_photo)
            if len(matches) == 0:
                continue
            person_id = parts[1]
            with open('move.jsonl', 'a') as f:
                for match in matches:
                    new_path = f"CHNG_{person_id}_{count}.jpg"
                    count += 1
                    final_path = os.path.join(photo_folder, new_path)
                    match['new_path'] = final_path
                    if not os.path.exists(match['match_photo']):
                        
                        continue
                    shutil.move(match['match_photo'], final_path)
                    f.write(json.dumps(match) + '\n')
 
def comp(feat1, feat2):
    sim = rec.compute_sim(feat1, feat2)
    return sim
                

def compare_video(video_path, threshold=0.44):
    src_folder = "/Users/odms/Documents/aditya_ws/Photos"

    # Load database embeddings
    img_paths = []
    person_ids = []
    all_feats = []

    with open('final_embeddings_adaface_r101.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            path = data['path']
            feats = np.array(data['feat'])
            img_data = path.split('/')[-1]
            parts = img_data.split('_')
            person_no = parts[1]
            img_no = parts[2].split('.')[0]

            img_paths.append(path)
            all_feats.append(feats)
            person_ids.append(person_no)

    features_matrix = np.stack(all_feats).astype('float32')
    dim = features_matrix.shape[1]
    faiss.normalize_L2(features_matrix)
    index = faiss.IndexFlatIP(dim)
    index.add(features_matrix)

    print(f'Loaded {len(img_paths)} embeddings')

    # Open video stream
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detect faces
        bboxes, kpss = detector.autodetect(frame, max_num=5)
        if bboxes.shape[0] == 0:
            continue

        for kps in kpss:
            feat = rec.get(frame, kps)
            if feat is None:
                continue

            feat = np.array(feat, dtype='float32').reshape(1, -1)
            faiss.normalize_L2(feat)
            D, I = index.search(feat, 1)

            sim = float(D[0][0])
            match_index = int(I[0][0])

            matched_img = img_paths[match_index]
            match_person_no = matched_img.split('_')[1]

            # Define your known vs unknown person IDs
            known_persons = ['0005', '0010', '0015']  # example: persons present in final_embeddings
            status = ""

            if sim >= threshold:
                if match_person_no in known_persons:
                    status = f"✅ Matched {match_person_no} (sim: {sim:.2f})"
                else:
                    status = f"❌ Wrong Match to {match_person_no} (sim: {sim:.2f})"
            else:
                if match_person_no in known_persons:
                    status = f"❌ Missed known {match_person_no} (sim: {sim:.2f})"
                else:
                    status = f"✅ Correctly rejected (sim: {sim:.2f})"

            print(status)

        # FPS calculation
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f'Processed {frame_count} frames at {fps:.2f} FPS')

    cap.release()
    total_time = time.time() - start_time
    avg_fps = frame_count / total_time
    print(f'Total frames: {frame_count}, Average FPS: {avg_fps:.2f}')


# def compare_video_detailed(video_path, threshold=0.44, output_path="output_video.mp4"):
#     src_folder = "/Users/odms/Documents/aditya_ws/Photos"

#     # Load embeddings DB
#     img_paths = []
#     person_ids = []
#     all_feats = []

#     num_embeddings = 0
#     with open('final_embeddings_adaface_ov.jsonl', 'r') as f:
#         for line in f:
#             data = json.loads(line)
#             path = data['path']
#             feats = np.array(data['feat'])
#             img_data = path.split('/')[-1]
#             parts = img_data.split('_')
#             person_no = parts[1]
#             img_paths.append(path)
#             all_feats.append(feats)
#             person_ids.append(person_no)
#             if num_embeddings == 100000:
#                 break
#             num_embeddings += 1

#     features_matrix = np.stack(all_feats).astype('float32')
#     dim = features_matrix.shape[1]
#     faiss.normalize_L2(features_matrix)
#     index = faiss.IndexFlatIP(dim)
#     # index = faiss.IndexHNSWFlat(dim, 32)
#     index.add(features_matrix)

#     # m = 16      # number of subvectors
#     # nbits = 8   # bits per subvector
#     # index = faiss.IndexIVFPQ(faiss.IndexFlatIP(dim), dim, 1024, m, nbits)
#     # index.train(features_matrix)
#     # index.add(features_matrix)
#     # index.nprobe = 10

#     print(f'Loaded {len(img_paths)} embeddings')

#     # Define known persons present in DB
#     known_persons = ['0005', '0010', '0015']  # Adjust as per your case

#     # Open video
#     cap = cv2.VideoCapture(video_path)
#     frame_count = 0
#     start_time = time.time()

#     # Video writer setup
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     # Stats counters
#     total_faces = 0
#     matched_known = 0
#     matched_unknown = 0
#     missed_known = 0
#     correctly_rejected = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
        
#         bboxes, kpss = detector.autodetect(frame, max_num=5)
#         if bboxes.shape[0] == 0:
#             out.write(frame)
#             continue

#         total_faces += bboxes.shape[0]

#         for i in range(bboxes.shape[0]):
#             kps = kpss[i]
#             box = bboxes[i].astype(int)
#             feat = rec.get(frame, kps)
#             if feat is None:
#                 continue

#             feat = np.array(feat, dtype='float32').reshape(1, -1)
#             faiss.normalize_L2(feat)
#             D, I = index.search(feat, 1)

#             sim = float(D[0][0])
#             match_index = int(I[0][0])
#             matched_img = img_paths[match_index]
#             match_person_no = matched_img.split('_')[1]

#             # Determine match status # TODO: Correct This
#             if sim >= threshold:
#                 if match_person_no in known_persons:
#                     status = f"sim ({sim:.2f})"
#                     color = (0, 255, 0)
#                     matched_known += 1
#                 else:
#                     status = f"sim ({sim:.2f})" # TODO: Correct This
#                     color = (0, 255, 0)
#                     matched_unknown += 1
#             else:
#                 if match_person_no in known_persons:
#                     status = f"sim ({sim:.2f})"
#                     color = (0, 0, 255)
#                     missed_known += 1
#                 else:
#                     status = f"sim ({sim:.2f})"
#                     color = (0, 0, 255)
#                     correctly_rejected += 1

#             # Draw bounding box and label
#             cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
#             cv2.putText(frame, status, (box[0], box[1] - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         out.write(frame)

#         if frame_count % 30 == 0:
#             elapsed = time.time() - start_time
#             fps_now = frame_count / elapsed
#             print(f'Processed {frame_count} frames at {fps_now:.2f} FPS')

#     cap.release()
#     out.release()

#     total_time = time.time() - start_time
#     avg_fps = frame_count / total_time

#     print(f"Total frames: {frame_count}, Total faces: {total_faces}, Average FPS: {avg_fps:.2f}")
#     print(f"Matched known faces: {matched_known}")
#     print(f"Matched unknown faces (false positive): {matched_unknown}")
#     print(f"Missed known faces (false negative): {missed_known}")
#     print(f"Correctly rejected unknown faces: {correctly_rejected}")
#     print(f"Output video saved to: {output_path}")


def compare_video_detailed(video_path, threshold=0.44, output_path="output_video.mp4"):
    src_folder = "/Users/odms/Documents/aditya_ws/Photos"

    # Load embeddings DB
    img_paths, person_ids, all_feats = [], [], []
    num_embeddings = 0

    with open('final_embeddings_adaface_ov_fp16.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            path = data['path']
            feats = np.array(data['feat'])
            img_data = path.split('/')[-1]
            parts = img_data.split('_')
            person_no = parts[1]
            img_paths.append(path)
            all_feats.append(feats)
            person_ids.append(person_no)
            if num_embeddings == 100000:
                break
            num_embeddings += 1

    features_matrix = np.stack(all_feats).astype('float32')
    dim = features_matrix.shape[1]
    faiss.normalize_L2(features_matrix)
    
    index = faiss.IndexFlatIP(dim)
    index.add(features_matrix)

    # nlist=1024
    # index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist, faiss.METRIC_INNER_PRODUCT)
    # index.train(features_matrix)
    # index.add(features_matrix)
    # index.nprobe = 10
    
    # index = faiss.IndexHNSWFlat(dim, 32)
    # index.add(features_matrix)
    # dist_sq_thr = 2 - 2 * threshold  # For cosine similarity, threshold is 1 - cosine distance

    print(f'Loaded {len(img_paths)} embeddings')

    known_persons = ['0005', '0010', '0015']

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    start_time = time.time()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_faces, matched_known, matched_unknown, missed_known, correctly_rejected = 0, 0, 0, 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        frame_start = time.time()

        # Face Detection Timing
        fd_start = time.time()
        bboxes, kpss = detector.autodetect(frame, max_num=5)
        fd_time = time.time() - fd_start

        if bboxes.shape[0] == 0:
            out.write(frame)
            continue

        total_faces += bboxes.shape[0]

        for i in range(bboxes.shape[0]):
            kps = kpss[i]
            box = bboxes[i].astype(int)

            # Face Recognition Timing
            fr_start = time.time()
            feat = rec.get(frame, kps)
            fr_time = time.time() - fr_start

            if feat is None:
                continue

            feat = np.array(feat, dtype='float32').reshape(1, -1)
            faiss.normalize_L2(feat)

            # FAISS Search Timing
            search_start = time.time()
            D, I = index.search(feat, 1)
            search_time = time.time() - search_start

            sim = float(D[0][0])
            match_index = int(I[0][0])
            matched_img = img_paths[match_index]
            match_person_no = matched_img.split('_')[1]

            # Determine match status # TODO: Correct This
            if sim >= threshold:
                if match_person_no in known_persons:
                    status = f"sim ({sim:.2f})"
                    color = (0, 255, 0)
                    matched_known += 1
                else:
                    status = f"sim ({sim:.2f})" # TODO: Correct This
                    color = (0, 255, 0)
                    matched_unknown += 1
            else:
                if match_person_no in known_persons:
                    status = f"sim ({sim:.2f})"
                    color = (0, 0, 255)
                    missed_known += 1
                else:
                    status = f"sim ({sim:.2f})"
                    color = (0, 0, 255)
                    correctly_rejected += 1


            # Draw timing
            draw_start = time.time()
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, status, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            draw_time = time.time() - draw_start

        # Encoding/Output Timing
        encode_start = time.time()
        out.write(frame)
        encode_time = time.time() - encode_start

        frame_total_time = time.time() - frame_start

        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            avg_fps_now = frame_count / elapsed
            print(f"[{frame_count} frames] FPS: {avg_fps_now:.2f} | "
                  f"FD: {fd_time:.3f}s | FR: {fr_time:.3f}s | "
                  f"Search: {search_time:.3f}s | Draw: {draw_time:.3f}s | "
                  f"Encode: {encode_time:.3f}s | Frame Total: {frame_total_time:.3f}s")

    cap.release()
    out.release()

    total_time = time.time() - start_time
    avg_fps = frame_count / total_time

    print(f"\n==== Final Stats ====")
    print(f"Total frames: {frame_count}, Total faces: {total_faces}, Average FPS: {avg_fps:.2f}")
    print(f"Matched known faces: {matched_known}")
    print(f"Matched unknown faces (false positive): {matched_unknown}")
    print(f"Missed known faces (false negative): {missed_known}")
    print(f"Correctly rejected unknown faces: {correctly_rejected}")
    print(f"Output video saved to: {output_path}")


def compare_video_live_display(video_path, threshold=0.44, output_path="output_video.mp4"):
    src_folder = "/Users/odms/Documents/aditya_ws/Photos"

    # Load embeddings DB
    img_paths = []
    person_ids = []
    all_feats = []

    num_embeddings = 0
    with open('final_embeddings_adaface_ov.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            path = data['path']
            feats = np.array(data['feat'])
            img_data = path.split('/')[-1]
            parts = img_data.split('_')
            person_no = parts[1]
            img_paths.append(path)
            all_feats.append(feats)
            person_ids.append(person_no)
            if num_embeddings == 50:
                break
            num_embeddings += 1

    features_matrix = np.stack(all_feats).astype('float32')
    dim = features_matrix.shape[1]
    faiss.normalize_L2(features_matrix)
    index = faiss.IndexFlatIP(dim)
    index.add(features_matrix)

    print(f'Loaded {len(img_paths)} embeddings')

    # Define known persons present in DB
    known_persons = ['0005', '0010', '0015']

    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    start_time = time.time()

    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Stats counters
    total_faces = 0
    matched_known = 0
    matched_unknown = 0
    missed_known = 0
    correctly_rejected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        bboxes, kpss = detector.autodetect(frame, max_num=5)
        if bboxes.shape[0] > 0:
            total_faces += bboxes.shape[0]

            for i in range(bboxes.shape[0]):
                kps = kpss[i]
                box = bboxes[i].astype(int)
                feat = rec.get(frame, kps)
                if feat is None:
                    continue

                feat = np.array(feat, dtype='float32').reshape(1, -1)
                faiss.normalize_L2(feat)
                D, I = index.search(feat, 1)

                sim = float(D[0][0])
                match_index = int(I[0][0])
                matched_img = img_paths[match_index]
                match_person_no = matched_img.split('_')[1]

                # Determine match status
                if sim >= threshold:
                    if match_person_no in known_persons:
                        status = f"sim ({sim:.2f})"
                        color = (0, 255, 0)
                        matched_known += 1
                    else:
                        status = f"sim ({sim:.2f})"
                        color = (0, 255, 0)
                        matched_unknown += 1
                else:
                    if match_person_no in known_persons:
                        status = f"sim ({sim:.2f})"
                        color = (0, 0, 255)
                        missed_known += 1
                    else:
                        status = f"sim ({sim:.2f})"
                        color = (0, 0, 255)
                        correctly_rejected += 1

                # Draw bounding box and label
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, status, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # FPS display overlay
        elapsed = time.time() - start_time
        fps_now = frame_count / elapsed
        fps_text = f"FPS: {fps_now:.2f}"
        cv2.putText(frame, fps_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        # Show live video window
        cv2.imshow("Face Recognition Live", frame)

        # Write frame to output video
        out.write(frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            print("Quit by user.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    total_time = time.time() - start_time
    avg_fps = frame_count / total_time

    print(f"\n--- Performance Stats ---")
    print(f"Total frames: {frame_count}, Total faces: {total_faces}, Average FPS: {avg_fps:.2f}")
    print(f"Matched known faces: {matched_known}")
    print(f"Matched unknown faces (false positive): {matched_unknown}")
    print(f"Missed known faces (false negative): {missed_known}")
    print(f"Correctly rejected unknown faces: {correctly_rejected}")
    print(f"Output video saved to: {output_path}")

           
if __name__ == '__main__':
    import time as time
    # args = parse_args()
    # output = func(args)
    # print('sim: %.4f, message: %s'%(output[0], output[1]))
    # compare4(0.7)
    # compare4(0.65)
    # compare4(0.6)
    # compare4(0.55)
    # compare4(0.5)
    #go_through()
    st = time.time()
    # compare4(0.44)
    # compare_video('/Users/odms/Documents/aditya_ws/fr_code/classroom.gif', threshold=0.45)
    compare_video_detailed('videos/input/39837-424360872_small.mp4', threshold=0.4, output_path='videos/output/39837-424360872_small_04_npu_wo_blur_200.mp4')
    # compare_video_live_display('/Users/odms/Documents/aditya_ws/fr_code/videos/input/39837-424360872_small.mp4', threshold=0.45, output_path='/Users/odms/Documents/aditya_ws/fr_code/videos/output/output_video_walking_with_match_live_1920-1080_45.mp4')
    print(f"Time taken: {time.time() - st} seconds")
    # st2 = time.time()
    # compare4(0.43)
    # print(f"Time taken: {time.time() - st2} seconds")
    # st3 = time.time()
    # compare4(0.42)
    # print(f"Time taken: {time.time() - st3} seconds")
    # st4 = time.time()
    # compare4(0.41)
    # print(f"Time taken: {time.time() - st4} seconds")
    # compare4(0.35)
    # compare4(0.3)
    # compare4(0.25)
    # compare4(0.2)

    # compare4(0.45)
    # compare4(0.43)
    # compare4(0.41)
    # compare4(0.6)
    # compare4(0.3)
    
    

