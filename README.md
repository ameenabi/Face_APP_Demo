# Face Parsing App
A simple Streamlit app that performs face segmentation using a pre-trained BiSeNet model and allows you to change hair color interactively.

🚀 Features
1. Upload a face image
2. Segment different facial regions
3. Visualize face parsing results
4. Change hair color using a color picker

## 📂 Project Structure

```
Face_APP_Demo/
├── app.py                    # Main Streamlit app  
├── model_org.py              # BiSeNet model definition  
├── model_files/  
│   └── 1.5L_iterations.pth   # Pretrained model weights (Not committed to GitHub)  
├── demo_images_data/         # Folder to upload and store images  
├── segmentation_outputs/     # Stores output segmented images  
├── requirements.txt          # List of required Python packages  
└── README.md                 # Project documentation  
```

🛠️ How to Run
1. Clone the repository
2. git clone https://github.com/your-username/Face_APP_Demo.git
3. cd Face_APP_Demo
4. Install dependencies
5. pip install -r requirements.txt

Run the app
streamlit run streamlit_app.py

🖼️ Example
Upload a face image (.jpg, .png)
See side-by-side:
Original image
Segmented image
Use the color picker to recolor the hair



# Face Parsing Model Conversion (PyTorch → TFLite)

This project provides a step-by-step pipeline to convert a **PyTorch segmentation model** (`BiSeNet`) into a **TensorFlow Lite (TFLite)** model for deployment on edge/mobile devices.

---

## 📂 Project Structure
```
├── model_files/
│ └── 1.5L_iterations.pth # Pretrained PyTorch model checkpoint
├── output_model.onnx # ONNX exported model
├── output_model_NHWC.onnx # ONNX model in NHWC format
├── tensorflow_out/ # SavedModel directory
├── face_parsing_model.tflite # Final TFLite model
├── model_org.py # Contains BiSeNet architecture
└── torch_to_tflite.py # Conversion script
```

---

## 🧩 Dependencies

Install required packages using Python 3.10 for compatibility:

1. pip install torch torchvision
2. pip install tensorflow==2.11.0
3. pip install tensorflow-addons==0.19.0

## 🛠️ Conversion Pipeline

1. Load PyTorch Model and Export to ONNX
2. Convert ONNX to NHWC Format
3. Convert ONNX to TensorFlow SavedModel
4. Test TensorFlow Model Output
5. Convert TensorFlow Model to TFLite
