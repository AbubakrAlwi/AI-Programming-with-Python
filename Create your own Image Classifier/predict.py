import torch
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F
import argparse
import numpy as np
from PIL import Image
import json
import os
import random




def Command_line():
    # Create the parser and include arguments
    parser = argparse.ArgumentParser(description="Predict flower name from an image")
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k',type= int ,default= 3, help=' provide the top k most probable classes.')
    parser.add_argument('--filepath', dest='filepath', default='flowers/test/57/image_08127.jpg' , help='the path to a single image') 
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action = 'store_true', help='will use the GPU to run the model if the cuda is accessible.')

    return parser.parse_args()

def read_flower_name(classes,category_names):
    if (category_names is not None):
        cat_file = category_names 
        cat_to_name = json.loads(open(cat_file).read())
        names = [cat_to_name[str(c)] for c in classes]
        return names
    return None

def load_checkpoint(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    
    architecture = checkpoint['architecture']
    valid_architectures = {
        'densenet121': models.densenet121,
        'vgg16': models.vgg16,
        'alexnet': models.alexnet }

    if architecture in valid_architectures:
        model = valid_architectures[architecture](pretrained=True)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}. Choose from {list(valid_architectures.keys())}.")
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    model.epochs = checkpoint['epochs']
    model.optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model 

def process_image(image_path):
    img = Image.open(image_path)
    # Resize while preserving aspect ratio
    if img.size[0] > img.size[1]:
        img.thumbnail((9999999, 256))
    else:
        img.thumbnail((256, 9999999))
    
    # Center crop to 224x224
    left = (img.width - 224) / 2
    top  = (img.height - 224) / 2
    right = left + 224
    bottom = top + 224
    img = img.crop((left, top, right, bottom))
    
    # Convert to numpy array and normalize
    np_image = np.array(img) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Reorder dimensions for PyTorch
    np_image = np_image.transpose((2, 0, 1))
    
    return torch.from_numpy(np_image).type(torch.FloatTensor)



def predict( model, args):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Allocate GPU if requested and available
    device = "cuda" if args.gpu == True and torch.cuda.is_available() else "cpu"
    topk= args.top_k
    model.to(device)
    model.eval()
    
    
    # Prepare the image
    img = process_image(args.filepath).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
    
    # Convert to numpy
    top_p = top_p.cpu().numpy().flatten()
    top_class = top_class.cpu().numpy().flatten()
    
    # Convert indices to classes
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_labels = [idx_to_class[i] for i in top_class]
    
    return top_p, top_labels




def main():
    # Get user-defined arguments
    args = Command_line() 

    # Load the saved check point
    model=load_checkpoint(args.checkpoint)

    # predict an image with path provided by user
    probs, classes = predict( model, args)
    
    # get flowers real names 
    flower_real_name = read_flower_name(classes,args.category_names)

    # print predictions
    i=0
    while i < len(classes):
        print("#{: <3} {: <25} Prob: {:.2f}%".format(i+1, flower_real_name[i], probs[i]*100))
        i += 1 
if __name__ == '__main__':
    main()


