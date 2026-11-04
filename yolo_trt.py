"""YOLO26 detection straight on TensorRT: no PyTorch, so ~300 MB instead of ~760 MB.

Reads an Ultralytics-exported .engine (its header holds the class names) and returns
(name, confidence, (x1, y1, x2, y2)) in original frame pixels. Handles both export forms:
end2end (1, 300, 6) output is already the final box list; the classic (1, 4+classes, 8400)
output is raw candidates that still need non-maximum suppression (NMS).
"""
import json

import cv2
import numpy as np
import tensorrt as trt

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart


def _check(result):
    err, *rest = result
    if int(err) != 0:
        raise RuntimeError(f"CUDA error {err}")
    return rest[0] if len(rest) == 1 else rest


class TrtDetector:
    def __init__(self, engine_path):
        data = open(engine_path, "rb").read()
        meta_len = int.from_bytes(data[:4], "little")  # Ultralytics: 4-byte length + JSON
        meta = json.loads(data[4:4 + meta_len])
        self.names = {int(k): v for k, v in meta["names"].items()}
        self.end2end = meta.get("end2end", True)
        self.engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(data[4 + meta_len:])
        if self.engine is None:
            raise RuntimeError(f"Could not load {engine_path}; re-export it on this Jetson")
        self.context = self.engine.create_execution_context()
        self.stream = _check(cudart.cudaStreamCreate())
        self.buffers = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            host = np.zeros(tuple(self.engine.get_tensor_shape(name)),
                            dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
            device = _check(cudart.cudaMalloc(host.nbytes))
            self.context.set_tensor_address(name, device)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            self.buffers[name] = (host, device, is_input)
        self.input = next(host for host, _, is_input in self.buffers.values() if is_input)
        self.output = next(host for host, _, is_input in self.buffers.values() if not is_input)
        self.size = self.input.shape[2]

    def detect(self, frame, conf=0.5, iou=0.7):
        # Letterbox: scale to fit the square input, pad the rest with grey (as Ultralytics does).
        h, w = frame.shape[:2]
        scale = min(self.size / h, self.size / w)
        nh, nw = round(h * scale), round(w * scale)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        self.input[0] = canvas[:, :, ::-1].transpose(2, 0, 1) / 255.0  # BGR->RGB, HWC->CHW

        to_device = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        to_host = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        for host, device, is_input in self.buffers.values():
            if is_input:
                _check(cudart.cudaMemcpyAsync(device, host.ctypes.data, host.nbytes, to_device, self.stream))
        self.context.execute_async_v3(self.stream)
        for host, device, is_input in self.buffers.values():
            if not is_input:
                _check(cudart.cudaMemcpyAsync(host.ctypes.data, device, host.nbytes, to_host, self.stream))
        _check(cudart.cudaStreamSynchronize(self.stream))

        if self.end2end:
            rows = self.output[0]
            boxes, scores, classes = rows[:, :4], rows[:, 4], rows[:, 5].astype(int)
            keep = np.flatnonzero(scores >= conf)
        else:
            # Each column is one candidate: centre x, centre y, width, height, then class scores.
            candidates = self.output[0].T
            classes = candidates[:, 4:].argmax(1)
            scores = candidates[np.arange(len(candidates)), 4 + classes]
            cx, cy, bw, bh = candidates[:, :4].T
            boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
            mask = scores >= conf
            boxes, scores, classes = boxes[mask], scores[mask], classes[mask]
            # Merge overlapping boxes of the same class, keeping the most confident one.
            # Shifting each class far apart stops plain NMS merging boxes of different classes.
            shifted = boxes + (classes * 4 * self.size)[:, None]
            xywh = np.column_stack([shifted[:, :2], shifted[:, 2:] - shifted[:, :2]])
            keep = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), conf, iou),
                            dtype=int).reshape(-1)

        detections = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            box = ((x1 - left) / scale, (y1 - top) / scale, (x2 - left) / scale, (y2 - top) / scale)
            detections.append((self.names[int(classes[i])], float(scores[i]),
                               tuple(int(round(min(max(v, 0), lim))) for v, lim in zip(box, (w, h, w, h)))))
        return detections
