import argparse
from model_funcs import * # bad practice but whatevs
from helper_funcs import * 

parser = argparse.ArgumentParser(prog='TRAIN', description='Train a neural network')
parser.add_argument('data_dir', type=str, help='select directory of training and testing data')
parser.add_argument('--save_dir', default='checkpoint_new' ,type=str, help='save checkpoint to different directory (default: save to home directory)')
parser.add_argument('--arch', default='vgg16', type=str, help='neural network architecture')
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--hidden_units', type=int, default=512)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--gpu', dest='gpu', action='store_true', help='use gpu for training')
args = parser.parse_args()

device = 'cuda' if args.gpu else 'cpu'

for key,value in vars(args).items():
    print(key, ': ', value)
print('device: ',device)

data_dir = set_image_directories(args.data_dir)
data_transforms = set_transforms()
datasets = load_datasets(data_dir, data_transforms)
dataloaders = create_dataloaders(datasets)

model = create_base_model(args.arch, args.hidden_units, datasets['train'].class_to_idx)
criterion, optimizer, scheduler = set_training_params(model, args.learning_rate)
model = train_model(model, dataloaders['train'], dataloaders['valid'], criterion, optimizer, scheduler, epochs=args.epochs, device=device, print_every=40)
save_model(model, optimizer, args.save_dir, args.arch)