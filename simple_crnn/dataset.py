
import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class CRNNDataset(Dataset):
    def __init__(self, root, split='train', img_height=32, img_width=128, char_map=None):
        self.root = os.path.join(root, split)
        self.img_height = img_height
        self.img_width = img_width
        self.data = []
        
        # Check for label.txt
        label_file = os.path.join(self.root, 'label.txt')
        
        if os.path.exists(label_file):
            print(f"Loading labels from {label_file}")
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        filename = parts[0]
                        label = " ".join(parts[1:]) 
                        img_path = os.path.join(self.root, filename)
                        if os.path.exists(img_path):
                            self.data.append({'path': img_path, 'label': label})
        
        # Fallback to filenames
        if not self.data:
            self.image_paths = sorted(glob.glob(os.path.join(self.root, '*')))
            for p in self.image_paths:
                if os.path.isdir(p): continue
                if p.endswith('.txt'): continue
                
                filename = os.path.basename(p)
                label = self.get_label_from_filename(filename)
                self.data.append({'path': p, 'label': label})
        
        # Build Vocab
        if char_map is None:
            all_chars = set()
            for item in self.data:
                all_chars.update(item['label'])
            self.chars = sorted(list(all_chars))
            # 0 is reserved for CTC blank
            self.char2int = {c: i + 1 for i, c in enumerate(self.chars)}
            self.int2char = {i + 1: c for i, c in enumerate(self.chars)}
        else:
            self.char2int = char_map
            # Reverse map
            self.int2char = {i: c for c, i in char_map.items()}
            self.chars = sorted(list(self.char2int.keys()))
            
        self.vocab_size = len(self.chars)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def get_label_from_filename(self, filename):
        base = os.path.splitext(filename)[0]
        if '_' in base:
            parts = base.split('_')
            if parts[-1].isdigit():
                return "_".join(parts[:-1])
        return base

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['path']
        label = item['label']

        img = cv2.imread(img_path)
        if img is None: 
             img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

        # Basic Resize
        img = cv2.resize(img, (self.img_width, self.img_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_tensor = self.transform(img)
        
        # Encode label
        encoded = [self.char2int[c] for c in label if c in self.char2int]
        encoded_label = torch.tensor(encoded, dtype=torch.long)
        label_len = torch.tensor(len(encoded_label), dtype=torch.long)
        
        return img_tensor, encoded_label, label_len, label
