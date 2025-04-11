import torch
import torchvision
from torch import nn, optim
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from torch.utils import data
import matplotlib
import matplotlib.pyplot as plt
import argparse
from PIL import Image
import numpy as np
import os, random
import json
import time



# Create a parser argument as a function
def Command_line():
    
    parser = argparse.ArgumentParser(description="Training a neural network on a given dataset")
    
     # Path for the dataset folder which contain subfolders training , validation and testing .
    parser.add_argument('--data_directory',dest="data_directory",default = 'flowers', help='The directory path containing the train, test and valid folders, each of which has its own collection of train, test and valid image data sets.')
    
    # Set directory to save checkpoints
    parser.add_argument('--save_dir', dest="save_dir", action="store", type=str ,default = '/home/workspace/ImageClassifier' , help='directory path to save checkpoints.')
    
     # Network architecture.
    parser.add_argument('--arch', choices=['densenet121','alexnet','vgg16'], default='densenet121', help='model architecture, for instance: densenet121' ) 
    
    # Hyperparameters
    parser.add_argument('--hidden_units', dest='hidden_units', type=int,default= 512, help='Number of hidden layers, example:512')
    parser.add_argument('--epochs', dest='epochs', default = 5 ,type=int, help='epoch numbers. example: 20')
    parser.add_argument('--learning_rate', dest='learning_rate',type=float, default= 0.001, help='Learning rate value, example: 0.002')
    parser.add_argument('--dropout', dest = "dropout", action = "store",type=float, default = 0.2, help='Dropout value, example: 0.2')
    
    # Use GPU for training .
    parser.add_argument('--gpu', dest='gpu', action='store_true', help="Use this parameter if you intend to use CUDA to train the model on the GPU" )
    
    # Execute parse_args()
    return parser.parse_args()



def data_path(args):
    train_dir = args.data_directory + '/train'
    test_dir = args.data_directory + '/test'
    valid_dir = args.data_directory + '/valid'
    return train_dir,valid_dir,test_dir

def data_transformation(train_dir,valid_dir,test_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
    val_dataset = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
    valid_loader  = torch.utils.data.DataLoader(val_dataset, batch_size = 64,  shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 64,   shuffle = True)
    data_loader =[train_loader,valid_loader,test_loader]
    return data_loader , train_dataset.class_to_idx


def define_architecture(args):
 

    # Allocate GPU if requested and available
    device = "cuda" if args.gpu == True and torch.cuda.is_available() else "cpu"

    # Validate architecture name
    if args.arch not in models.__dict__:
        raise ValueError(f"Invalid architecture name: {args.arch}. Please choose a valid model from torchvision.models.")

    # Load pretrained model
    model = getattr(models, args.arch)(pretrained=True)

    # Freeze feature extraction layers
    for param in model.parameters():
        param.requires_grad = False

    # Determine the feature size 
    if args.arch == 'alexnet':
        input_features = model.classifier[1].in_features
    elif args.arch == 'vgg16':
        input_features = model.classifier[0].in_features
    elif args.arch == 'densenet121':
        input_features = model.classifier.in_features 
    else:
        raise ValueError(f"Unable to determine input feature size for {args.arch}")

    # Define a custom classifier
    model.classifier = nn.Sequential(
        nn.Linear(input_features, args.hidden_units),
        nn.ReLU(),
        nn.Dropout(p=args.dropout),
        nn.Linear(args.hidden_units, args.hidden_units // 2),  # Integer division
        nn.ReLU(),
        nn.Dropout(p=args.dropout),
        nn.Linear(args.hidden_units // 2 , 102),  # 102 flower classes
        nn.LogSoftmax(dim=1)
    )

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)


    return model, criterion, optimizer, device


def train_model(args,train_loader, valid_loader,model_parameters):
    # Initialize model, loss function, optimizer, and device
    model, criterion, optimizer, device = model_parameters
    # Training parameters
    steps = 0
    print_every = 10  
    start_time = time.time()

    # Move model to the appropriate device
    model.to(device)
    print("Training started...\n")
    
    # Training loop
    for epoch in range(args.epochs):
        running_loss = 0
        #model.train()  # Set to training mode
        
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            # Backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print status every `print_every` steps
            if steps % print_every == 0 or steps == 1 or steps == len(train_loader):
                valid_loss = 0
                accuracy = 0
                model.eval()  # Turn off dropout for validation

                # Validation loop (no gradient tracking)
                with torch.no_grad():
                    for inputs_val, labels_val in valid_loader:
                        inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                        logps_val = model(inputs_val)
                        batch_loss = criterion(logps_val, labels_val)
                        
                        valid_loss += batch_loss.item()
                        
                        # Compute accuracy
                        ps = torch.exp(logps_val)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels_val.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{args.epochs}.. "
                      f"Train Loss: {running_loss / print_every:.3f}.. "
                      f"Valid Loss: {valid_loss / len(valid_loader):.3f}.. "
                      f"Valid Accuracy: {accuracy / len(valid_loader):.3f}")

                running_loss = 0
                model.train()  # Return to training mode

    # Training time
    total_time = time.time() - start_time
    print("\n Total training time: {:.0f}m {:.0f}s".format(total_time // 60, total_time % 60))
    print("\n Training complete!")

    return model

def accuracy_test(criterion,device,model, test_loader):
    model.eval()
    test_loss = 0
    accuracy = 0
    print('validation is Starting')
    with torch.no_grad():
        for inputs_test, labels_test in test_loader:
            inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)

            logps_test = model(inputs_test)
            batch_loss = criterion(logps_test, labels_test)
            test_loss += batch_loss.item()
        
            ps_test = torch.exp(logps_test)
            top_p, top_class = ps_test.topk(1, dim=1)
            equals = top_class == labels_test.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    print('\n Validation is finished \n')
    print(f"Test Loss: {test_loss/len(test_loader):.3f}.. " f"Test Accuracy: {accuracy/len(test_loader):.3f}")
    
def save_checkpoint(model, optimizer, args,class_to_idx):

    # Extract Input Shape Dnamically
    input_features = model.classifier[0].in_features

    model.class_to_idx = class_to_idx

    checkpoint = {'input_size': input_features,
                 'output_size': 102,
                'architecture': args.arch,
                 'classifier' : model.classifier,
               'learning_rate': args.learning_rate,
                      'epochs': args.epochs,
                'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict(),
                   'optimizer': optimizer.state_dict(),
             }

    torch.save(checkpoint, os.path.join(args.save_dir, 'checkpoint.pth'))
    print("file is saved")            



def main():
    # Get user-defined arguments
    args = Command_line() 
    # getting data path
    train_dir,valid_dir,test_dir= data_path(args)
    # Load data
    dataloaders , class_to_idx = data_transformation(train_dir,valid_dir,test_dir)  # Unpack data loaders
    train_loader, valid_loader ,test_loader = dataloaders  # Unpack data loaders

    # Initialize model, loss function, optimizer, and device
    model, criterion, optimizer, device = define_architecture(args)
    model_parameters = model, criterion, optimizer, device 
    # Initiate training 
    trained_model =  train_model(args,train_loader, valid_loader,model_parameters)
    # save the new trained model 
    save_checkpoint(trained_model, optimizer, args, class_to_idx)
    # examine accuracy
    accuracy_test(criterion,device,trained_model, test_loader)

   
    
   
   
    

if __name__ == '__main__':
    main()

















