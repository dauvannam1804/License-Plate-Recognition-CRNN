
import os
import torch
import cv2
import argparse
from torchvision import transforms
from model import CRNN
from dataset import CRNNDataset 
from utils import decode_greedy

def get_vocab(dataset_root):
    # Quick scan to build vocab
    train_dataset = CRNNDataset(dataset_root, split='train')
    return train_dataset.char2int, train_dataset.int2char, train_dataset.vocab_size

def infer(image_path, model_path, dataset_root):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Setup Encoder & Model
    print("Building vocabulary from dataset...")
    char2int, int2char, vocab_size = get_vocab(dataset_root)
    
    model = CRNN(vocab_size=vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Preprocess Image
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Could not read image."
        
    img = cv2.resize(img, (128, 32)) # Match training
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device) # (1, C, H, W)
    
    # 3. Inference
    with torch.no_grad():
        preds = model(img_tensor) # (T, 1, C)
        log_probs = torch.nn.functional.log_softmax(preds, dim=2)
        decoded = decode_greedy(log_probs, int2char)
        
    return decoded[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to image')
    parser.add_argument('--model', type=str, default='../checkpoints/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='../dataset_final', help='Root of dataset to build vocab')
    
    args = parser.parse_args()
    
    result = infer(args.image, args.model, args.dataset)
    print(f"Prediction: {result}")
