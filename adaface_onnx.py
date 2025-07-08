# -*- coding: utf-8 -*-
# @Organization  : Adapted from insightface.ai
# @Author        : Adapted from Jia Guo
# @Time          : 2023-10-01
# @Function      : AdaFace ONNX inference

import numpy as np
import cv2
import onnx
import onnxruntime
import face_align

__all__ = [
    'AdaFaceONNX',
]

class AdaFaceONNX:
    def __init__(self, model_file=None, session=None):
        assert model_file is not None
        self.model_file = model_file
        self.session = session
        self.taskname = 'recognition'
        find_sub = False
        find_mul = False
        model = onnx.load(self.model_file)
        graph = model.graph
        for nid, node in enumerate(graph.node[:8]):
            if node.name.startswith('Sub') or node.name.startswith('_minus'):
                find_sub = True
            if node.name.startswith('Mul') or node.name.startswith('_mul'):
                find_mul = True
        # Set normalization for AdaFace (input scaled to [-1, 1])
        input_mean = 127.5
        input_std = 127.5
        self.input_mean = input_mean
        self.input_std = input_std
        if self.session is None:
            # Try CUDA, fallback to CPU if not available
            try:
                self.session = onnxruntime.InferenceSession(self.model_file, providers=['CUDAExecutionProvider'])
            except:
                self.session = onnxruntime.InferenceSession(self.model_file, providers=['CPUExecutionProvider'])
        input_cfg = self.session.get_inputs()[0]
        input_shape = input_cfg.shape
        input_name = input_cfg.name
        self.input_size = tuple(input_shape[2:4][::-1])
        self.input_shape = input_shape
        outputs = self.session.get_outputs()
        output_names = [out.name for out in outputs]
        self.input_name = input_name
        self.output_names = output_names
        # Select the first output as the primary feature embedding
        self.output_shape = outputs[0].shape
        print(f"Model outputs: {self.output_names}, using primary output: {self.output_names[0]}")

    def prepare(self, ctx_id, **kwargs):
        if ctx_id < 0 or 'CUDAExecutionProvider' not in self.session.get_providers():
            self.session.set_providers(['CPUExecutionProvider'])

    def get(self, img, kps):
        aimg = face_align.norm_crop(img, landmark=kps, image_size=self.input_size[0])
        embedding = self.get_feat(aimg).flatten()
        return embedding

    def compute_sim(self, feat1, feat2):
        from numpy.linalg import norm
        feat1 = feat1.ravel()
        feat2 = feat2.ravel()
        sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))
        return sim

    def get_feat(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        input_size = self.input_size
        # AdaFace expects BGR input, so do not swap R and B channels
        blob = cv2.dnn.blobFromImages(imgs, 1.0 / self.input_std, input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=False)
        # Use only the first output (feature embedding)
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out

    def forward(self, batch_data):
        # Assuming batch_data is in BGR order
        blob = (batch_data - self.input_mean) / self.input_std
        # Use only the first output
        net_out = self.session.run(self.output_names, {self.input_name: blob})[0]
        return net_out