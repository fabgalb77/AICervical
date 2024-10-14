from LocalizationModel import CoordRegressionNetwork
import torch
import dsntnn
from pydicom import dcmread
import numpy as np
import cv2 as cv
import albumentations as A
from albumentations.pytorch import ToTensorV2
import streamlit as st
from copy import deepcopy
from spine import *
from PIL import Image

def get_transforms(image_shape):
    transform = A.Compose([A.Resize(image_shape[0], image_shape[1], cv.INTER_CUBIC), # our input size can be 600px
                               ToTensorV2()])
    return transform

def normalize(image, norm_type):
    if norm_type == None:
        image = image
    elif norm_type == 'mean':       
        image = (image - image.mean()) / torch.std(image)
    return image

@st.cache()
def load_model(arch, n_locations_global, n_locations_local, pretrained, n_ch, n_blocks, device):
    model = CoordRegressionNetwork(arch, n_locations_global, n_locations_local, pretrained, n_ch, n_blocks).to(device)      
    w1 = torch.load('./src/model/dict1.pth')
    w2 = torch.load('./src/model/dict2.pth')
    w1.update(w2)
    model.load_state_dict(w1)
    return model

@st.cache()
def predict(img, model):

    img_name = img.name
    image_shape = (1024, 1024)

    transform = get_transforms(image_shape) 
    # =============================================================================
    kernel_size = (int(image_shape[0] / 256), int(image_shape[1] / 256))

    model.kernel_size = kernel_size

    pixel_size = -1.

    if img_name.endswith('.dcm'):
        dicom = dcmread(img, force=True)
        image = dicom.pixel_array
        pixel_size = float(dicom.PixelSpacing[0])
        print(pixel_size)

        zeroed_image = image - image.min()
        scaled_image = (np.maximum(zeroed_image, 0) / zeroed_image.max()) * 255.0
        scaled_image = np.uint8(scaled_image)
        org_image = scaled_image
    else:
        file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
        org_image = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)

    img_dc = deepcopy(org_image)
    img_dc = cv2.cvtColor(img_dc, cv2.COLOR_GRAY2RGB)
    
    org_shape = org_image.shape
    transformation_dict = transform(image = org_image)

    image = transformation_dict['image'].div(255.)
    image = normalize(image, 'mean')

    image = image.to(device)

    with torch.no_grad():
        coords_pred, heatmaps, coords_local = model(image.unsqueeze(dim = 0))

    h_org, w_org = org_shape[:2]

    h, w = image_shape

    h_org, w_org = int(h_org), int(w_org)
    h, w = int(h), int(w)

    coords_retransformed_pred = dsntnn.normalized_to_pixel_coordinates(coords_pred, size = image_shape)
    coords_retransformed_pred_local = dsntnn.normalized_to_pixel_coordinates(coords_local, size = image_shape)

    dims = torch.stack((torch.tensor(w_org / w), torch.tensor(h_org / h)))

    coords_original_pred = coords_retransformed_pred.cpu() * dims        
    coords_original_pred_local = coords_retransformed_pred_local.cpu() * dims

    coords_original_pred = coords_original_pred.squeeze()
    coords_original_pred_local = coords_original_pred_local.squeeze()

    #reorder coordinates to have them according to the annotation
    ind_reorder = [0, 1, 4, 5, 2, 3, 8, 9, 6, 7, 12, 13, 10, 11, 16, 17, 14, 15, 20, 21, 18, 19, 22, 23]
    ind_correct = [22, 23, 26, 27, 28]

    coords_original_pred_local = coords_original_pred_local[ind_reorder]
    #coords_original_pred_local = np.concatenate((coords_original_pred_local, coords_original_pred[ind_correct]))
    coords_original_pred_local = np.concatenate((coords_original_pred_local[:-2], coords_original_pred[ind_correct[:2]], coords_original_pred_local[-2:], coords_original_pred[ind_correct[2:]]))

    #models_pred_local = np.concatenate((models_pred_local[:, :-2], models_pred[:, ind_correct[:2]], models_pred_local[:, -2:], models_pred[:, ind_correct[2:]]), axis = 1)

    sp = spine(coords_original_pred_local, pixel_size)
    img_landmarks = sp.plot_all(img_dc)
    img_c0c2 = sp.plot_angle(img_dc, [(26, 27), (25, 24)], 'C0-C2: {:.1f} deg'.format(sp.c0c2()))
    img_c2c7 = sp.plot_angle(img_dc, [(24, 25), (5, 4)], 'C2-C7: {:.1f} deg'.format(sp.c2c7()))
    img_c2c6 = sp.plot_angle(img_dc, [(24, 25), (9, 8)], 'C2-C6: {:.1f} deg'.format(sp.c2c6()))
    img_t1_slope = sp.plot_angle_horizontal(img_dc, (0, 1), 'T1 slope: {:.1f} deg'.format(sp.t1_slope()))
    img_c7_slope = sp.plot_angle_horizontal(img_dc, (2, 3), 'C7 slope: {:.1f} deg'.format(sp.c7_slope()))
    img_sva = sp.plot_sva(img_dc)
    img_cranial_tilt = sp.plot_cranial_tilt(img_dc)
    img_cervical_tilt = sp.plot_cervical_tilt(img_dc)
    img_redlund = sp.plot_redlund(img_dc)

    images = {
        "landmarks" : img_landmarks,
        "T1 slope" : img_t1_slope,
        "C7 slope" : img_c7_slope, 
        "SVA" : img_sva,
        "cranial tilt" : img_cranial_tilt,
        "cervical tilt": img_cervical_tilt,
        "C0-C2" : img_c0c2,
        "C2-C6" : img_c2c6,
        "C2-C7" : img_c2c7,
        "Redlund-Johnell dist." : img_redlund
    }

    parameters = sp.get_parameters()

    return images, parameters, sp
    

def check_password():
    """Returns `True` if the user had a correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        try:
            if (
                st.session_state["username"] in st.secrets["passwords"]
                and st.session_state["password"]
                == st.secrets["passwords"][st.session_state["username"]]
            ):
                st.session_state["password_correct"] = True
                del st.session_state["password"]  # don't store username + password
                del st.session_state["username"]
            else:
                st.session_state["password_correct"] = False
        except:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input("Username", on_change=password_entered, key="username")
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("User not known or password incorrect")
        return False
    else:
        # Password correct.
        return True

if __name__ == '__main__':
    if check_password():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        arch = 'hg2'
        n_ch = 1
        n_blocks = 1
        pretrained = True
        
        n_locations_global = 29
        n_locations_local = 4
        warmup_epoch = 20

        model = load_model(arch, n_locations_global, n_locations_local, pretrained, n_ch, n_blocks, device)
        model.eval() 

        logo = Image.open('src/logo_small.png')
        st.image(logo)

        file = st.file_uploader('Upload a cervical radiograph', accept_multiple_files=False, label_visibility='hidden')

        if file:
            images, parameters, sp = predict(file, model)
            #st.image(img_dc)
            print('T1 slope: {}\n'.format(sp.t1_slope()))
            print('C2-C7: {}\n'.format(sp.c2c7()))
            print('Cranial tilt: {}\n'.format(sp.cranialTilt()))
            print('Cervical tilt: {}\n'.format(sp.cervicalTilt()))
            sva_px, sva_mm = sp.sva()
            print('SVA: {} {}\n'.format(sva_px, sva_mm))
            print('C0-C2: {}\n'.format(sp.c0c2()))
            print('C2-C6: {}\n'.format(sp.c2c6()))
            print('C7 slope: {}\n'.format(sp.c7_slope()))
            rd_px, rd_mm = sp.redlund()
            print('Redlund: {} {}\n'.format(rd_px, rd_mm))

            st.table(parameters.style.format({"value": "{:.1f}"}))

            images_option = st.selectbox('Select a radiographic parameter', ('landmarks', 'T1 slope', 'C7 slope', 'SVA', 'cranial tilt', 'cervical tilt', 'C0-C2', 'C2-C6', 'C2-C7', 'Redlund-Johnell dist.'))
            st.image(images[images_option])

            st.subheader("Disclaimer")
            st.write("The authors do not guarantee the accuracy, reliability or currency of the information provided with this tool. Any errors in the information that are brought to our attention will be corrected as soon as possible. We reserve the right to change at any time without notice any information stored in the tool. The authors accept no liability for any loss or damage a person suffers because that person has directly or indirectly relied on any information provided by this tool.")