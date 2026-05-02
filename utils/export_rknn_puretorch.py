from pathlib import Path

import torch
import onnx
import onnxslim
from rknn.api import RKNN

class ModelConverter:
    def __init__(self, model_path):
        self.model_path = model_path
        self.target_platform = {0:"rk3588"}

    def convert_torch2onnx(self, simplify):

        model = torch.load(self.model_path, weights_only=False)

        # Ensure to disable training layers
        model.eval()
        # Example input should be tuple of tensors  
        input_example = (torch.randn(1, 3, 192, 320),)
        onnx_file = str(Path(self.model_path).with_suffix(".onnx"))
        torch.onnx.export(model,
                          input_example,
                          onnx_file,
                          input_names=["input_image"],
                          output_names=["output"],
                          opset_version=19,
                          do_constant_folding=False,
                          dynamo=False,
                          verbose=False,
                        )
        '''dynamic_axes={
                "input_image": {0: "batch_size",
                                2: "height",
                                3: "width"},
                "output": {0: "batch_size",
                            2: "anchors"}
                            }'''
        onnx_model = onnx.load(onnx_file)
        if simplify:
            onnx_model = onnxslim.slim(onnx_model)
        onnx.save(onnx_model, onnx_file)
        return onnx_file


    def convert_onnx2rknn(self, onnx_file):
        rknn = RKNN()
        rknn.config(
            target_platform=self.target_platform.get(0),
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
        )
        '''
            quantized_dtype='asymmetric_quantized-8',
            quantized_algorithm='mmse',
            optimization_level=3
            '''
        ret = rknn.load_onnx(onnx_file)
        if not ret:
            pass
        rknn.build(do_quantization=False, rknn_batch_size=1)
        rknn_file = str(Path(self.model_path).with_suffix(".rknn"))
        rknn.export_rknn(rknn_file)

model_converter = ModelConverter("/home/erfan/projects/sarbazienv/workspace/rknn_opt/models/pure_torch/vd_640_v11.pt")
onnx_file = model_converter.convert_torch2onnx(True)
model_converter.convert_onnx2rknn(onnx_file)

