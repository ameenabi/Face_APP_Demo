from model_org import BiSeNet
import torch


n_classes = 19
pt_model_path = "/Users/shaik.ameenabi/Documents/Git_Projects/Face_APP_Demo/model_files/1.5L_iterations.pth"


model = BiSeNet(n_classes=n_classes)


model.load_state_dict(torch.load(pt_model_path, map_location='cpu'))
# batch_size, channels, height, width
sample_input = torch.rand((1, 3, 512, 512))


torch.onnx.export(model, sample_input, "output_model.onnx", opset_version=12, input_names=['input'] ,output_names=['output'] )

from scc4onnx import order_conversion


order_conversion(
   input_onnx_file_path='./output_model.onnx',
   output_onnx_file_path='output_model_NHWC.onnx',
   input_op_names_and_order_dims={"input": [0,2,3,1]}
)


import onnx
onnx_model = onnx.load('output_model_NHWC.onnx')

from onnx_tf.backend import prepare
tf_rep = prepare(onnx_model)
tf_model_path = "/home/ameena/Documents/Face_APP/Lip_Segmentation/face-parsing.PyTorch/tensorflow_out/"
tf_rep.export_graph(tf_model_path)


# Testing the tensorflow model ##


import tensorflow as tf
model = tf.saved_model.load(tf_model_path)
model.trainable = False
input_tensor = tf.random.uniform([1, 512, 512, 3])
out = model(**{'input': input_tensor})
print(out.shape)

# Testing the tensorflow model ##

import tensorflow as tf
model = tf.saved_model.load(tf_model_path)
model.trainable = False
input_tensor = tf.random.uniform([1, 512, 512, 3])
out = model(**{'input': input_tensor})
print(out.shape)

## converting tensorflow model to tflite ##


import tensorflow as tf
tf_model_path = "/home/ameena/Documents/Face_APP/Lip_Segmentation/face-parsing.PyTorch/tensorflow_out/"
converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
tflite_model = converter.convert()
with open("./face_parsing_model.tflite", 'wb') as f:
   f.write(tflite_model)