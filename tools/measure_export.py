import numpy as np
import onnxruntime as ort
import torch

import tensorrt as trt
# import pycuda.driver as cuda
# import pycuda.autoinit
from collections import namedtuple, OrderedDict


class ONNXInfer():
    def __init__(self, onnx_model_path):
        self.session = ort.InferenceSession(onnx_model_path)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [i.name for i in self.session.get_outputs()]

    def run(self, input_data):
        return self.session.run(
            self.output_names,
            {self.input_names[0]: input_data.astype(np.float32)}
        )

class TensorRTInfer():
    def __init__(self, tensorrt_model_path):
        self.tensorrt_model_path = tensorrt_model_path
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, namespace="")
        with open(tensorrt_model_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.bindings = OrderedDict()
        for index in range(self.engine.num_bindings):
            name = self.engine.get_tensor_name(index)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = tuple(self.engine.get_tensor_shape(name))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).cuda()
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = self.engine.create_execution_context()

        # warm up
        name = self.engine.get_tensor_name(0)
        dtype = trt.nptype(self.engine.get_tensor_dtype(name))
        shape = tuple(self.engine.get_tensor_shape(name))
        input_data = torch.from_numpy(np.random.rand(*shape).astype(dtype)).cuda()
        for _ in range(10):
            self.binding_addrs[name] = int(input_data.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            self.bindings[self.engine.get_tensor_name(1)].data

    def run(self, input_data):
        self.binding_addrs[self.engine.get_tensor_name(0)] = int(torch.from_numpy(input_data.astype(np.float16)).cuda().data_ptr())
        flag = self.context.execute_v2(list(self.binding_addrs.values()))
        if flag == False:
            print(f"TensorRT model {self.tensorrt_model_path} Inference Failed!!!")
        outputs = []
        for index in range(self.engine.num_bindings):
            if index == 0:
                continue
            output = self.bindings[self.engine.get_tensor_name(index)].data
            outputs.append(
                output.cpu().numpy()
            )
        return outputs


def main(engine = "onnx"):
    print(f"val precision of {engine} ...")
    ############################ encoder.camera ###################################################
    input_data_path = "assets/encoder.camera.input.txt"
    output_depth_data_path = "assets/encoder.camera.depth.output.txt"
    output_feats_data_path = "assets/encoder.camera.feats.output.txt"
    input_data = np.loadtxt(input_data_path).reshape(1, 6, 3, 256, 704)
    output_depth_data = np.loadtxt(output_depth_data_path).reshape(6, 118, 32, 88)
    output_feats_data = np.loadtxt(output_feats_data_path).reshape(6, 32, 88, 80)

    if engine == "onnx":
        onnx_model_path = "runs/seg_camera_only_resnet50/onnx_fp16/camera.backbone.onnx"
        camera_onnx_infer = ONNXInfer(onnx_model_path)
        output_feats, output_depth = camera_onnx_infer.run(input_data)

        atol_threshold1, atol_threshold2 = 1e-3, 1e-2

    if engine == "tensorrt":
        trt_model_path = "runs/seg_camera_only_resnet50/build/onnx_fp16/camera.backbone.plan"
        camera_trt_infer = TensorRTInfer(trt_model_path)
        outputs = camera_trt_infer.run(input_data)
        output_depth, output_feats = outputs[0], outputs[1]
        atol_threshold1, atol_threshold2 = 1e-2, 1e-1

    if np.allclose(output_depth_data, output_depth, atol=atol_threshold1):
        print(f"export encoder.camera.depth {engine} correct 🤗🤗🤗, threshold: {atol_threshold1}")
    else:
        print(f"export encoder.camera.depth {engine} incorrect 😥😥😥, threshold: {atol_threshold1}")

    if np.allclose(output_feats_data, output_feats, atol=atol_threshold2):
        print(f"export encoder.camera.feats {engine} correct 🤗🤗🤗, threshold: {atol_threshold2}")
    else:
        print(f"export encoder.camera.feats {engine} incorrect 😥😥😥, threshold: {atol_threshold2}")

    ################################ vtransform.downsample ################################
    input_data_path = "assets/vtransform.downsample.input.txt"
    output_data_path = "assets/vtransform.downsample.output.txt"
    input_data = np.loadtxt(input_data_path).reshape(1, 80, 256, 256)
    output_data = np.loadtxt(output_data_path).reshape(1, 80, 128, 128)
    if engine == "onnx":
        onnx_model_path = "runs/seg_camera_only_resnet50/onnx_fp16/camera.vtransform.onnx"
        vtransform_onnx_infer = ONNXInfer(onnx_model_path)
        output_feats = vtransform_onnx_infer.run(input_data)[0]
        atol_threshold = 1e-2
    if engine == "tensorrt":
        trt_model_path = "runs/seg_camera_only_resnet50/build/onnx_fp16/camera.vtransform.plan"
        vtransform_trt_infer = TensorRTInfer(trt_model_path)
        output_feats = vtransform_trt_infer.run(input_data)
        atol_threshold = 1e-1

    if np.allclose(output_data, output_feats, atol=atol_threshold):
        print(f"export vtransform.downsample {engine} correct 🤗🤗🤗, threshold: {atol_threshold}")
    else:
        print(f"export vtransform.downsample {engine} incorrect 😥😥😥, threshold: {atol_threshold}")

    ################################ decoder #################################################
    input_data_path = "assets/decoder.input.txt"
    output_data_path = "assets/decoder.output.txt"
    input_data = np.loadtxt(input_data_path).reshape(1, 80, 128, 128)
    output_data = np.loadtxt(output_data_path).reshape(1, 256, 128, 128)

    if engine == "onnx":
        onnx_model_path = "runs/seg_camera_only_resnet50/onnx_fp16/fuser.onnx"
        decoder_onnx_infer = ONNXInfer(onnx_model_path)
        output_feats = decoder_onnx_infer.run(input_data)[0]
        atol_threshold = 1e-2
    if engine == "tensorrt":
        trt_model_path = "runs/seg_camera_only_resnet50/build/onnx_fp16/fuser.plan"
        decoder_trt_infer = TensorRTInfer(trt_model_path)
        output_feats = decoder_trt_infer.run(input_data)
        atol_threshold = 1e-1

    if np.allclose(output_data, output_feats, atol=atol_threshold):
        print(f"export decoder {engine} correct 🤗🤗🤗, threshold: {atol_threshold}")
    else:
        print(f"export decoder {engine} incorrect 😥😥😥, threshold: {atol_threshold}")

    ################################# head.map.classifier ####################################
    input_data_path = "assets/head.map.classifier.input.txt"
    output_data_path = "assets/head.map.classifier.output.txt"
    input_data = np.loadtxt(input_data_path).reshape(1, 256, 200, 200)
    output_data = np.loadtxt(output_data_path).reshape(1, 6, 200, 200)

    if engine == "onnx":
        onnx_model_path = "runs/seg_camera_only_resnet50/onnx_fp16/head.map.onnx"
        decoder_onnx_infer = ONNXInfer(onnx_model_path)
        output_feats = decoder_onnx_infer.run(input_data)[0]
        atol_threshold = 1e-3
    if engine == "tensorrt":
        trt_model_path = "runs/seg_camera_only_resnet50/build/onnx_fp16/head.map.plan"
        head_trt_infer = TensorRTInfer(trt_model_path)
        output_feats = head_trt_infer.run(input_data)
        atol_threshold = 1e-2

    if np.allclose(output_data, output_feats, atol=atol_threshold):
        print(f"export head.map.classifier {engine} correct 🤗🤗🤗, threshold: {atol_threshold}")
    else:
        print(f"export head.map.classifier {engine} incorrect 😥😥😥, threshold: {atol_threshold}")

if __name__ == "__main__":
    engine = "onnx"
    main(engine)
    engine = "tensorrt"
    main(engine)