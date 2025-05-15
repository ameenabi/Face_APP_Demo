

# Face APP

# Face Parsing Model Conversion (PyTorch → TFLite)

This project provides a step-by-step pipeline to convert a **PyTorch segmentation model** (`BiSeNet`) into a **TensorFlow Lite (TFLite)** model for deployment on edge/mobile devices.

---

## 📂 Project Structure

├── model_files/
│ └── 1.5L_iterations.pth # Pretrained PyTorch model checkpoint
├── output_model.onnx # ONNX exported model
├── output_model_NHWC.onnx # ONNX model in NHWC format
├── tensorflow_out/ # SavedModel directory
├── face_parsing_model.tflite # Final TFLite model
├── model_org.py # Contains BiSeNet architecture
└── torch_to_tflite.py # Conversion script


---

## 🧩 Dependencies

Install required packages using Python 3.10 for compatibility:

pip install torch torchvision
pip install tensorflow==2.11.0
pip install tensorflow-addons==0.19.0

## 🛠️ Conversion Pipeline

1. Load PyTorch Model and Export to ONNX
2. Convert ONNX to NHWC Format
3. Convert ONNX to TensorFlow SavedModel
4. Test TensorFlow Model Output
5. Convert TensorFlow Model to TFLite
