import os
import sys
src_path = os.path.split(os.getcwd())[0]
sys.path.insert(0, src_path)

import json
import random
import glob

import numpy as np
import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from PIL import Image

import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import streamlit as st
import matplotlib.pyplot as plt

import clip_model as clip
from clip_model import _transform
from model import CLIPGeneral
import zeroshot_data


def load_image(image_file):
	img = Image.open(image_file)
	return img

def load_cloob_model():
    # Set the GPU
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    checkpoint_path = "../checkpoints/cloob_rn50_yfcc_epoch_28.pt"
    # checkpoint_path = "/content/drive/My Drive/Colab Notebooks/checkpoints/epoch_30.pt"

    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    model_config_file = os.path.join('../cloob/src/training/model_configs/', checkpoint['model_config_file'])
    # model_config_file = os.path.join(src_path, 'content/drive/My Drive/Colab Notebooks/training/model_configs/RN50.json')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device is ", device)
    print('Loading model from', model_config_file)

    with open(model_config_file, 'r') as f:
        model_info = json.load(f)
        print('model_info: ', model_info)

    model_cloob = CLIPGeneral(**model_info)

    preprocess= _transform(model_cloob.visual.input_resolution, is_train=False)
    if not torch.cuda.is_available():
        model_cloob.float()
    else:
        model_cloob.to(device)
    sd = checkpoint["state_dict"]
    sd = {k[len('module.'):]: v for k, v in sd.items()}
    if 'logit_scale_hopfield' in sd:
        sd.pop('logit_scale_hopfield', None)

    model_cloob.load_state_dict(sd)
    return model_cloob, preprocess, device

def zero_shot_classifier(model, classnames, templates, device):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).to(device) #tokenize
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights

def run(model, classifier, dataloader, device, accuracy_metric):
    with torch.no_grad():
        all_logits = []
        all_targets = []
        for images, target in tqdm(dataloader):
            # images = images.to(device)
            # target = target.to(device)

            # predict
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            logits = image_features @ classifier
            print("Complete")

            all_logits.append(logits.cpu())
            all_targets.append(target.cpu())

        print("Done running")

        all_logits = torch.cat(all_logits).numpy()
        all_targets = torch.cat(all_targets).numpy()

        acc = accuracy_metric(all_targets, all_logits.argmax(axis=1)) * 100.0
        return acc

def predict_image(classifier, img, preprocess, model_cloob, device):
    original_images = []
    processed_images = []

    with Image.open(img) as im:
        original_images.append(im)
        processed_images.append(preprocess(im))
    processed_images = torch.stack(processed_images)

    model_cloob.eval()
    images = processed_images.to(device)
    with torch.no_grad():
        image_features = model_cloob.encode_image(images)
        image_features /= image_features.norm(dim=-1, keepdim=True)

    text_probs = (30.0 * image_features @ classifier).softmax(dim=-1)
    top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)

    top_probs = top_probs.cpu()
    top_labels = top_labels.cpu()
    text_probs = text_probs.cpu()

    return top_probs, top_labels, text_probs


def main():
    model_cloob, preprocess, device = load_cloob_model()

    data_path = "archive/animals"

    dataset = datasets.ImageFolder(data_path, transform=preprocess)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, num_workers=3)
    
    classnames = zeroshot_data.animal_classname
    prompt_templates = zeroshot_data.animal_templates

    model_cloob.eval()

    print("Calculating the text embeddings for all classes of the dataset", flush=True)
    classifier = zero_shot_classifier(model_cloob, classnames, prompt_templates, device)

    uploaded_files = st.file_uploader("Choose a file", type=["png","jpg","jpeg"], accept_multiple_files=True)
    
    for uploaded_file in uploaded_files:
        top_probs, top_labels, text_probs = predict_image(classifier, uploaded_file, preprocess, model_cloob, device)

        #  st.image(load_image(uploaded_file),width=250)
        st.image(uploaded_file, width=300)
        # st.write(classnames[top_labels[0].numpy()[0]])

        fig, ax = plt.subplots(figsize=(2, 2))
        y = np.arange(top_probs.shape[-1])
        ax.barh(y, top_probs[0], zorder=-1, color=[123/255.0,204/255.0,196/255.0,255/255.0])
        ax.invert_yaxis()
        ax.set_axisbelow(True)
        ax.set_yticks(y)
        ax.set_xlim([0,1])
        ax.yaxis.set_ticks_position('none') 
        ax.tick_params(axis="y")
        classnames_plot = classnames
        ax.set_yticklabels([classnames_plot[index] for index in top_labels[0].numpy()], x=0.05, zorder=1, horizontalalignment='left')

        st.pyplot(fig, use_container_width =False)




if __name__ == '__main__':
    main()
        
    

         
    

