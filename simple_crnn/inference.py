
import os
import torch
import cv2
import argparse
from torchvision import transforms
from model import CRNN
from dataset import CRNNDataset 
from utils import decode_greedy
import config as cfg

def get_vocab(dataset_root):
    # Quick scan to build vocab
    train_dataset = CRNNDataset(dataset_root, split='train')
    return train_dataset.char2int, train_dataset.int2char, train_dataset.vocab_size

def infer(image_path, model_path, dataset_root):
    device = cfg.DEVICE
    
    # 1. Setup Encoder & Model
    print("Building vocabulary from dataset...")
    char2int, int2char, vocab_size = get_vocab(dataset_root)
    
    model = CRNN(vocab_size=vocab_size, hidden_size=cfg.HIDDEN_SIZE).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 2. Preprocess Image
    img = cv2.imread(image_path)
    if img is None:
        return "Error: Could not read image."
        
    img = cv2.resize(img, (cfg.IMG_WIDTH, cfg.IMG_HEIGHT)) # Match training
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
    parser.add_argument('--model', type=str, default=os.path.join(cfg.CHECKPOINT_DIR, 'best_model.pth'), help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default=cfg.DATASET_ROOT, help='Root of dataset to build vocab')
    
    args = parser.parse_args()
    
    result = infer(args.image, args.model, args.dataset)
    print(f"Prediction: {result}")
