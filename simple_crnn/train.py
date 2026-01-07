
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset import CRNNDataset
from model import CRNN
from utils import decode_greedy, calculate_metrics, pad_collate
import config as cfg

def log_print(msg):
    print(msg)
    with open(cfg.LOG_FILE, 'a') as f:
        f.write(msg + '\n')
    
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    predictions = []
    targets = []
    
    for imgs, encoded_labels, label_lens, raw_labels in pbar:
        imgs = imgs.to(device)
        encoded_labels = encoded_labels.to(device)
        label_lens = label_lens.to(device)
        
        optimizer.zero_grad()
        preds = model(imgs) # (T, B, C)
        
        log_probs = torch.nn.functional.log_softmax(preds, dim=2)
        input_lengths = torch.full(size=(imgs.size(0),), fill_value=preds.size(0), dtype=torch.long).to(device)
        
        loss = criterion(log_probs, encoded_labels, input_lengths, label_lens)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
        # Decode for metrics
        decoded_preds = decode_greedy(log_probs, train_loader.dataset.int2char)
        predictions.extend(decoded_preds)
        targets.extend(raw_labels)
        
    avg_loss = total_loss / len(train_loader)
    char_acc, word_acc = calculate_metrics(predictions, targets)
    
    # log_print(f"Epoch {epoch} [Train] Loss: {avg_loss:.4f}, Char Acc: {char_acc:.4f}, Word Acc: {word_acc:.4f}")
    return avg_loss, char_acc, word_acc

def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        for imgs, encoded_labels, label_lens, raw_labels in pbar:
            imgs = imgs.to(device)
            encoded_labels = encoded_labels.to(device)
            label_lens = label_lens.to(device)
            
            preds = model(imgs)
            log_probs = torch.nn.functional.log_softmax(preds, dim=2)
            input_lengths = torch.full(size=(imgs.size(0),), fill_value=preds.size(0), dtype=torch.long).to(device)
            
            loss = criterion(log_probs, encoded_labels, input_lengths, label_lens)
            total_loss += loss.item()
            
            decoded_preds = decode_greedy(log_probs, val_loader.dataset.int2char)
            predictions.extend(decoded_preds)
            targets.extend(raw_labels)
            
    avg_loss = total_loss / len(val_loader)
    char_acc, word_acc = calculate_metrics(predictions, targets)
    
    # log_print(f"Epoch {epoch} [Val] Loss: {avg_loss:.4f}, Char Acc: {char_acc:.4f}, Word Acc: {word_acc:.4f}")
    return avg_loss, char_acc, word_acc

def main():
    # 1. Datasets
    # Load train first to determine vocab
    train_dataset = CRNNDataset(cfg.DATASET_ROOT, split='train', img_height=cfg.IMG_HEIGHT, img_width=cfg.IMG_WIDTH)
    # Pass char mapping to validation
    val_dataset = CRNNDataset(cfg.DATASET_ROOT, split='val', img_height=cfg.IMG_HEIGHT, img_width=cfg.IMG_WIDTH, char_map=train_dataset.char2int)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, collate_fn=pad_collate, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, collate_fn=pad_collate, num_workers=4)
    
    # 2. Model
    model = CRNN(vocab_size=train_dataset.vocab_size, hidden_size=cfg.HIDDEN_SIZE).to(cfg.DEVICE)
    
    # 3. Optimization
    criterion = nn.CTCLoss(blank=0, zero_infinity=True) 
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    
    best_loss = float('inf')
    
    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss, _, _ = train(model, train_loader, criterion, optimizer, cfg.DEVICE, epoch)
        val_loss, char_acc, word_acc = validate(model, val_loader, criterion, cfg.DEVICE, epoch)
        
        # Log combined metrics for plotting
        log_print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Char Acc: {char_acc:.4f}, Word Acc: {word_acc:.4f}")
        
        # Save best model based on LOSS
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(cfg.CHECKPOINT_DIR, 'best_model.pth'))
            log_print(f"Saved best model (Loss: {best_loss:.4f}).")
            
    log_print("Training Complete.")

if __name__ == '__main__':
    main()
