import argparse
from model_funcs import *
from helper_funcs import * 

def read_user_input():
    parser = argparse.ArgumentParser(prog='PREDICT', description='Predict an image')
    parser.add_argument('image_path', type=str, help='path to image')
    parser.add_argument('checkpoint_path', type=str, help='path to checkpoint for loading model') #default='checkpoint_new.pth', 
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--gpu', action='store_true', help='use gpu for inference')
    parser.add_argument('--category_names', type=str) 
    return parser.parse_args()

args = read_user_input()
device = 'cuda' if args.gpu else 'cpu'

for key,value in vars(args).items():
    print(key, ': ', value)

model = load_model(args.checkpoint_path)
top_probs, top_labels = predict(args.image_path, model, args.top_k, device=device)
if args.category_names:
    category_names = read_class_names(args.category_names)
    top_classes = [category_names[i] for i in top_labels]
    print( {'{class_} ({label})'.format(class_=class_, label=label): prob for class_, label, prob in zip(top_classes, top_labels, top_probs) } )
else:
    print( {label: prob for label, prob in zip(top_labels, top_probs) } )
  
