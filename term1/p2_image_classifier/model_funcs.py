# Imports here
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.optim import lr_scheduler
from collections import OrderedDict
from helper_funcs import process_image

def set_image_directories(data_dir):
    return {
            'train' : data_dir + '/train',
            'valid' : data_dir + '/valid',
            'test' : data_dir + '/test'
            }

def set_transforms():
    mu = (0.485, 0.456, 0.406)
    sigma = (0.229, 0.224, 0.225)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mu, sigma)
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mu, sigma)
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mu, sigma)
        ]),
    }
    return data_transforms

def load_datasets(data_dir, data_transforms):
    #Load datasets with ImageFolder
    return {x: datasets.ImageFolder(data_dir[x],   transform=data_transforms[x]) for x in ['train', 'valid', 'test']}

def create_dataloaders(datasets):
    return {x: torch.utils.data.DataLoader(datasets[x], batch_size=32, shuffle=True) for x in ['train', 'valid', 'test']}

def set_training_params(model, lr):
    # Set the cost function to NLLLoss
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr)
    # Set scheduler that decreases learning rate over time
    scheduler = lr_scheduler.StepLR (optimizer, step_size = 4, gamma = 0.1)
    return criterion, optimizer, scheduler

def create_base_model(arch, hidden_units, class_to_idx):
    print(arch)
    if arch =='vgg13':
        model = models.vgg13(pretrained=True)
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True) 
    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True) 
    else:
        raise AssertionError('Architecture not recognized: choose among vgg13, vgg16 and vgg19')
        
    model.classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    model.class_to_idx = class_to_idx
    return model

def save_model(model, optimizer, directory, arch):
    model.cpu()
    checkpoint = {
                  'arch' : arch, 
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'optimizer_state': optimizer.state_dict,
                 }
    torch.save(checkpoint, directory)
    
def load_model(checkpoint):
    # load checkpoint
    checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    # load model architecture
    model = models.vgg16(pretrained=True)
    # replace classifier
    model.classifier = checkpoint['classifier']
    # load class to idx mapping
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def validation(model, valloader, criterion, device):
    val_loss = 0
    accuracy = 0

    for images, labels in valloader:
        
        images, labels = images.to(device), labels.to(device)
        output = model.forward(images)
        val_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return val_loss, accuracy

def train_model(model, trainloader, valloader, criterion, optimizer, scheduler=None, epochs=25, device='cpu', print_every=40):
       
    # change to chosen device type
    model.to(device)

    for e in range(epochs):
        running_loss = 0
        steps = 0

        for ii, (inputs, labels) in enumerate(trainloader):
            model.train()
            steps += 1
            if scheduler:
                scheduler.step()
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    val_loss, accuracy = validation(model, valloader, criterion, device)
                
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Train Steps: {}... ".format(steps),
                      "Train Loss: {:.2f}".format(running_loss/print_every),
                      "Val Loss: {:.3f}.. ".format(val_loss/len(valloader)),
                      "Val Accuracy: {:.3f}".format(accuracy/len(valloader)))

                running_loss = 0
    return model

def predict(image_path, model, topk, device='cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    img = process_image(image_path)

    img = torch.from_numpy(img).type(torch.FloatTensor)
    img.unsqueeze_(0)

    model, img = model.to(device), img.to(device)

    with torch.no_grad():         
        model.eval()
        top_probs, top_labels = torch.exp(model.forward(img)).topk(topk)

    top_probs = top_probs.cpu().numpy()[0]
    top_labels = top_labels.cpu().numpy()[0]
    
    idx_to_label = {value: key for key, value in model.class_to_idx.items()}
    top_labels = [idx_to_label[i] for i in top_labels]
        
    return top_probs, top_labels

def evaluate_image(image_path, model, cat_to_name):   
    probs, labels, classes = predict(image_path, model, cat_to_name)

    fig, ax = plt.subplots(figsize=(4, 8))
    ax1 = plt.subplot(2,1,1)
    ax2 = plt.subplot(2,1,2)
    image = PIL.Image.open(image_path)
    image = resized(image, 255)
    image = centeredCrop(image, 224)
    correct_label = cat_to_name[image_path.split('/')[-2]]

    ax1.imshow(image)
    ax1.set_title(correct_label, fontsize=14)

    y_pos = np.arange(len(classes))
    ax2 = plt.barh(y_pos, probs)
    plt.gca().invert_yaxis()
    plt.yticks(y_pos, classes);
