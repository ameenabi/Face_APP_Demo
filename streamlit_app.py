import streamlit as st
import numpy as np
import cv2
from skimage.filters import gaussian
import torch
import torchvision.transforms as transforms
from model_org import BiSeNet
import os

# Global variables for color
global_r = 0
global_g = 0
global_b = 0
print("hi")
# Function to visualize parsing maps
def vis_parsing_maps(im, parsing_anno, save_path, stride, save_im):
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],[255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [100, 100, 100], [170, 255, 0],[0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],[0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    part_colors_dict = {i: color for i, color in enumerate(part_colors)}

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors_dict[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    if save_im:
        cv2.imwrite(save_path + "img_segmentation.png", vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        
    return vis_im

# Function to change hair color
def hair_color_chnge(img, mask, color):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, mask = cv2.threshold(mask, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
    green_hair = np.copy(img)
    green_hair[(mask==255).all(-1)] = color
    new_hair2 = cv2.addWeighted(green_hair.copy(), 0.2833333333333333, img, 0.7166666666666667, 0, green_hair.copy())
    return new_hair2

# Function to evaluate the model
def evaluate(image_path, model_path, device=None, respth='./segmentation_outputs/'):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.to(device)
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = cv2.resize(img_rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        norm_img = np.array(image)

        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        mask_image = np.zeros([512, 512, 3], np.uint8)
        hair_ind = np.where(parsing == 17)
        mask_image[hair_ind[0], hair_ind[1]] = 255

        vis_im = vis_parsing_maps(image, parsing, respth, stride=1, save_im=True)
        
    return vis_im, norm_img, mask_image

# Function to process the image
def process_image(input_image_path):
    model_path = './model_files/1.5L_iterations.pth'
    vis_im, norm_img, mask_image = evaluate(input_image_path, model_path)
    return vis_im, norm_img, mask_image

# Streamlit main function
if 'face_parsing' not in st.session_state:
    st.session_state.hair_color_pick = False
    st.session_state.hair_color_process = False

def main():
    st.title("Face Parsing APP")
    file_path = "./demo_images_data/"

    # Upload input image
    uploaded_file = st.file_uploader("Upload input image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            img_path = os.path.join(file_path, uploaded_file.name)
            with open(img_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            seg_img, norm_img, mask_image = process_image(img_path)
            cv2.imwrite("seg.png", seg_img)
            cv2.imwrite("org.png", cv2.cvtColor(norm_img, cv2.COLOR_RGB2BGR))
            
            # Display processed image in another column
            col1, col2 = st.columns(2)
            with col1:
                st.header("Uploaded Image")
                st.image("org.png", use_column_width=True)
                
            with col2:
                st.header("Segmented Image")
                st.image("seg.png", use_column_width=True)

            st.session_state.hair_color_pick = True

        except Exception as e:
            st.write("Upload Proper Image")
            st.write(f"Error: {e}")

    if st.session_state.hair_color_pick:
        st.header("Color Picker")    
        color = st.color_picker("", "#ff6347")  # Default color is optional
        rgb_color = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
        st.session_state.hair_color_process = True
        
    if st.session_state.hair_color_process:
        st.write("\n")
        if st.button("Change Color"):
            hair_changed = hair_color_chnge(norm_img, mask_image, [rgb_color[2], rgb_color[1], rgb_color[0]])
            cv2.imwrite("hair_style.png", hair_changed)
            st.image("hair_style.png", use_column_width=True)

if __name__ == "__main__":
    main()
