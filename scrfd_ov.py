# import numpy as np
# import openvino as ov
# import cv2


# def safe_squeeze(x, axis=0):
#     return np.squeeze(x, axis=axis) if x.shape[axis] == 1 else x


# class SCRFD_OV:
#     def __init__(self, model_path):
#         core = ov.Core()
#         self.model = core.read_model(model_path)
#         self.compiled_model = core.compile_model(self.model, 'CPU')
#         self.input_layer = self.compiled_model.input(0)
#         self.output_names = [o.get_any_name() for o in self.compiled_model.outputs]

#         self.fmc = 5  # For SCRFD 10G
#         self._feat_stride_fpn = [8, 16, 32, 64, 128]
#         self._num_anchors = 2
#         self.use_kps = True

#         self.det_thresh = 0.5
#         self.nms_thresh = 0.5

#         self.center_cache = {}
#         self.input_mean = 127.5
#         self.input_std = 128.0

#     def distance2bbox(self, points, distance):
#         x1 = points[:, 0] - distance[:, 0]
#         y1 = points[:, 1] - distance[:, 1]
#         x2 = points[:, 0] + distance[:, 2]
#         y2 = points[:, 1] + distance[:, 3]
#         return np.stack([x1, y1, x2, y2], axis=-1)

#     def distance2kps(self, points, distance):
#         preds = []
#         for i in range(0, distance.shape[1], 2):
#             px = points[:, 0] + distance[:, i]
#             py = points[:, 1] + distance[:, i+1]
#             preds.append(px)
#             preds.append(py)
#         return np.stack(preds, axis=-1).reshape(points.shape[0], -1, 2)

#     def nms(self, dets):
#         x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
#         areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#         order = scores.argsort()[::-1]
#         keep = []
#         while order.size > 0:
#             i = order[0]
#             keep.append(i)
#             xx1 = np.maximum(x1[i], x1[order[1:]])
#             yy1 = np.maximum(y1[i], y1[order[1:]])
#             xx2 = np.minimum(x2[i], x2[order[1:]])
#             yy2 = np.minimum(y2[i], y2[order[1:]])
#             w = np.maximum(0.0, xx2 - xx1 + 1)
#             h = np.maximum(0.0, yy2 - yy1 + 1)
#             inter = w * h
#             ovr = inter / (areas[i] + areas[order[1:]] - inter)
#             inds = np.where(ovr <= self.nms_thresh)[0]
#             order = order[inds + 1]
#         return keep

#     def forward(self, img, threshold):
#         scores_list = []
#         bboxes_list = []
#         kpss_list = []
#         input_size = tuple(img.shape[0:2][::-1])
#         blob = cv2.dnn.blobFromImage(img, 1.0 / self.input_std, input_size,
#                                     (self.input_mean, self.input_mean, self.input_mean),
#                                     swapRB=True)
#         result = self.compiled_model({'input': blob})

#         input_height = blob.shape[2]
#         input_width = blob.shape[3]
#         fmc = self.fmc

#         for idx, stride in enumerate(self._feat_stride_fpn):
#             scores = safe_squeeze(result[self.output_names[idx]])
#             bbox_preds = safe_squeeze(result[self.output_names[idx + fmc]]) * stride
#             if self.use_kps:
#                 kps_preds = safe_squeeze(result[self.output_names[idx + fmc * 2]]) * stride

#             height = input_height // stride
#             width = input_width // stride
#             K = height * width

#             key = (height, width, stride)
#             if key in self.center_cache:
#                 anchor_centers = self.center_cache[key]
#             else:
#                 anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
#                 anchor_centers = (anchor_centers * stride).reshape((-1, 2))
#                 if self._num_anchors > 1:
#                     anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
#                 if len(self.center_cache) < 100:
#                     self.center_cache[key] = anchor_centers

#             pos_inds = np.where(scores >= threshold)[0]
#             bboxes = distance2bbox(anchor_centers, bbox_preds)
#             pos_scores = scores[pos_inds]
#             pos_bboxes = bboxes[pos_inds]
#             scores_list.append(pos_scores)
#             bboxes_list.append(pos_bboxes)

#             if self.use_kps:
#                 kpss = distance2kps(anchor_centers, kps_preds)
#                 kpss = kpss.reshape((kpss.shape[0], -1, 2))
#                 pos_kpss = kpss[pos_inds]
#                 kpss_list.append(pos_kpss)

#         return scores_list, bboxes_list, kpss_list


#     def detect(self, img, input_size=(640, 640), thresh=None, max_num=0):
#         im_ratio = img.shape[0] / img.shape[1]
#         model_ratio = input_size[1] / input_size[0]
#         if im_ratio > model_ratio:
#             new_height = input_size[1]
#             new_width = int(new_height / im_ratio)
#         else:
#             new_width = input_size[0]
#             new_height = int(new_width * im_ratio)
#         det_scale = new_height / img.shape[0]

#         resized_img = cv2.resize(img, (new_width, new_height))
#         det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
#         det_img[:new_height, :new_width, :] = resized_img

#         det_thresh = thresh if thresh is not None else self.det_thresh
#         scores_list, bboxes_list, kpss_list = self.forward(det_img, det_thresh)

#         scores = np.hstack(scores_list)
#         bboxes = np.vstack(bboxes_list) / det_scale
#         kpss = np.vstack(kpss_list) / det_scale if self.use_kps else None

#         order = scores.argsort()[::-1]
#         pre_det = np.hstack((bboxes, scores[:, None]))[order]
#         keep = self.nms(pre_det)
#         det = pre_det[keep]
#         if self.use_kps:
#             kpss = kpss[order][keep]

#         if max_num > 0 and det.shape[0] > max_num:
#             area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
#             center = np.array([img.shape[1] // 2, img.shape[0] // 2])
#             offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - center[0], (det[:, 1] + det[:, 3]) / 2 - center[1]])
#             offset_dist = np.sum(offsets**2, axis=0)
#             values = area - offset_dist * 2.0
#             bindex = np.argsort(values)[::-1][:max_num]
#             det = det[bindex]
#             if kpss is not None:
#                 kpss = kpss[bindex]

#         return det, kpss

#     def autodetect(self, img, max_num=0):
#         bboxes, kpss = self.detect(img, input_size=(640, 640))
#         bboxes2, kpss2 = self.detect(img, input_size=(128, 128))
#         bboxes_all = np.vstack([bboxes, bboxes2])
#         kpss_all = np.vstack([kpss, kpss2]) if self.use_kps else None
#         keep = self.nms(bboxes_all)
#         det = bboxes_all[keep]
#         kpss = kpss_all[keep] if self.use_kps else None
#         if max_num > 0 and det.shape[0] > max_num:
#             area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
#             center = np.array([img.shape[1] // 2, img.shape[0] // 2])
#             offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - center[0], (det[:, 1] + det[:, 3]) / 2 - center[1]])
#             offset_dist = np.sum(offsets**2, axis=0)
#             values = area - offset_dist * 2.0
#             bindex = np.argsort(values)[::-1][:max_num]
#             det = det[bindex]
#             if kpss is not None:
#                 kpss = kpss[bindex]
#         return det, kpss


import openvino as ov
import numpy as np
import cv2
import os
import os.path as osp

def softmax(z):
    s = np.max(z, axis=1, keepdims=True)
    e_x = np.exp(z - s)
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def distance2bbox(points, distance):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)

def distance2kps(points, distance):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i+1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1).reshape(points.shape[0], -1, 2)

def safe_squeeze(x, axis=0):
    return np.squeeze(x, axis=axis) if x.shape[axis] == 1 else x

class SCRFD_OV:
    def __init__(self, model_file):
        self.model_file = model_file
        core = ov.Core()
        self.model = core.read_model(model_file)
        self.compiled_model = core.compile_model(self.model, 'CPU')
        self.output_names = [o.any_name for o in self.compiled_model.outputs]
        self.input_name = self.compiled_model.inputs[0].any_name
        # self.input_shape = self.compiled_model.inputs[0].shape
        input_shape = self.model.input().partial_shape
        if input_shape.is_dynamic:
            # Assume standard default SCRFD input
            self.input_shape = [1, 3, 640, 640]  # or [1, 3, 112, 112] based on your use
        else:
            self.input_shape = input_shape.to_shape()
        self.input_shape = [1, 3] + list(self.input_shape[2:4])
        self.input_size = tuple(self.input_shape[2:4][::-1])
        self.center_cache = {}
        self.det_thresh = 0.5
        self.nms_thresh = 0.5
        self.input_mean = 127.5
        self.input_std = 128.0
        self.use_kps = False
        self._num_anchors = 1
        num_outputs = len(self.output_names)
        if num_outputs == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
        elif num_outputs == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self.use_kps = True
        elif num_outputs == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif num_outputs == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True

    def forward(self, img, threshold):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(img, 1.0 / self.input_std, input_size,
                                     (self.input_mean, self.input_mean, self.input_mean),
                                     swapRB=True)
        result = self.compiled_model({self.input_name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]
        fmc = self.fmc

        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = safe_squeeze(result[self.output_names[idx]])
            bbox_preds = safe_squeeze(result[self.output_names[idx + fmc]]) * stride
            if self.use_kps:
                kps_preds = safe_squeeze(result[self.output_names[idx + fmc * 2]]) * stride

            height = input_height // stride
            width = input_width // stride

            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.tile(anchor_centers, (self._num_anchors, 1))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= threshold)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            if pos_scores.ndim > 1:
                pos_scores = pos_scores.squeeze()
            pos_bboxes = bboxes[pos_inds]
            if pos_scores.size == 0:
                continue  # skip empty detections for this stride
            else:
                pos_scores = np.asarray(pos_scores).flatten()  # ensure 1D
                pos_bboxes = np.asarray(pos_bboxes).reshape(-1, 4)
                scores_list.append(pos_scores)
                bboxes_list.append(pos_bboxes)
                if self.use_kps:
                    kpss = distance2kps(anchor_centers, kps_preds)
                    pos_kpss = kpss[pos_inds]
                    pos_kpss = np.asarray(pos_kpss).reshape(-1, self.num_keypoints, 2)
                    kpss_list.append(pos_kpss)

            # scores_list.append(pos_scores)
            # bboxes_list.append(pos_bboxes)

            # if self.use_kps:
            #     kpss = distance2kps(anchor_centers, kps_preds)
            #     pos_kpss = kpss[pos_inds]
            #     kpss_list.append(pos_kpss)

        return scores_list, bboxes_list, kpss_list

    def detect(self, img, input_size=None, thresh=None, max_num=0, metric='default'):
        # if not scores_list:
        #     return np.zeros((0, 5)), None

        input_size = self.input_size if input_size is None else input_size
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img
        det_thresh = thresh if thresh is not None else self.det_thresh

        scores_list, bboxes_list, kpss_list = self.forward(det_img, det_thresh)
        scores = np.concatenate(scores_list)
        bboxes = np.vstack(bboxes_list) / det_scale
        order = scores.argsort()[::-1]
        pre_det = np.hstack((bboxes, scores[:, None])).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = np.vstack(kpss_list)
            kpss = kpss[order]
            kpss = kpss[keep]
        else:
            kpss = None
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), axis=0)
            values = area if metric == 'max' else area - offset_dist_squared * 2.0
            bindex = np.argsort(values)[::-1][:max_num]
            det = det[bindex]
            if kpss is not None:
                kpss = kpss[bindex]
        return det, kpss

    def autodetect(self, img, max_num=0, metric='max'):
        bboxes, kpss = self.detect(img, input_size=(640, 640))
        bboxes2, kpss2 = self.detect(img, input_size=(128, 128))
        bboxes_all = np.concatenate([bboxes, bboxes2], axis=0)
        if self.use_kps:
            kpss_all = np.concatenate([kpss, kpss2], axis=0)
        else:
            kpss_all = None
        keep = self.nms(bboxes_all)
        det = bboxes_all[keep]
        if self.use_kps:
            kpss_all = kpss_all[keep]
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                 (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            offset_dist_squared = np.sum(np.power(offsets, 2.0), axis=0)
            values = area if metric == 'max' else area - offset_dist_squared * 2.0
            bindex = np.argsort(values)[::-1][:max_num]
            det = det[bindex]
            if self.use_kps:
                kpss_all = kpss_all[bindex]
        return det, kpss_all

    def nms(self, dets):
        thresh = self.nms_thresh
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep
