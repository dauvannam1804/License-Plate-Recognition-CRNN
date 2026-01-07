
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def pad_collate(batch):
    """
    Collate function to handle variable length targets.
    batch: list of tuples (img, content, content_len, label_str)
    """
    imgs, contents, content_lens, labels = zip(*batch)
    imgs = torch.stack(imgs, 0)
    content_lens = torch.stack(content_lens, 0)
    
    # Pad contents (encoded labels)
    max_len = max([len(c) for c in contents])
    padded_contents = torch.zeros((len(contents), max_len), dtype=torch.long)
    for i, c in enumerate(contents):
        padded_contents[i, :len(c)] = c
        
    return imgs, padded_contents, content_lens, labels

def decode_text(indices, int2char):
    """
    Decodes indices to text using int2char mapping.
    Ignoring indices not in mapping (like blanks/padding if mapped to 0 or ignored).
    """
    return "".join([int2char[i] for i in indices if i in int2char])

def decode_greedy(logits, int2char):
    """
    Decodes batch of logits using greedy CTC decoding.
    logits: (T, B, C)
    int2char: dict mapping int -> char
    """
    preds = logits.argmax(2) # (T, B)
    preds = preds.transpose(1, 0) # (B, T)
    preds = preds.cpu().numpy()
    
    decoded_texts = []
    for p in preds:
        # CTC Greedy Decode:
        # 1. Collapse repeats
        # 2. Remove blanks (0)
        decoded = []
        last = -1
        for val in p:
            if val != last and val != 0:
                decoded.append(val)
            last = val
        decoded_text = decode_text(decoded, int2char)
        decoded_texts.append(decoded_text)
    return decoded_texts

def calculate_metrics(predictions, targets):
    """
    predictions: list of strings
    targets: list of strings
    """
    assert len(predictions) == len(targets)
    
    total_chars = 0
    correct_chars = 0
    correct_words = 0
    
    for pred, target in zip(predictions, targets):
        # Word Accuracy
        if pred == target:
            correct_words += 1
            
        # Char Accuracy (Simple Match by Position)
        matches = 0
        min_len = min(len(pred), len(target))
        max_len = max(len(pred), len(target))
        
        for i in range(min_len):
            if pred[i] == target[i]:
                matches += 1
                
        correct_chars += matches
        total_chars += max_len
    
    char_acc = correct_chars / total_chars if total_chars > 0 else 0
    word_acc = correct_words / len(predictions) if len(predictions) > 0 else 0
    
    return char_acc, word_acc

def plot_training_log(log_file):
    """
    Reads the log file and plots training metrics.
    Expected format: "Epoch X: Train Loss: ..., Val Loss: ..., Char Acc: ..., Word Acc: ..."
    """
    epochs = []
    train_losses = []
    val_losses = []
    char_accs = []
    word_accs = []
    
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return

    with open(log_file, 'r') as f:
        for line in f:
            if "Train Loss:" in line and "Val Loss:" in line:
                try:
                    # Parse line like: 
                    # "Epoch 1: Train Loss: 12.5000, Val Loss: 10.2000, Char Acc: 0.1000, Word Acc: 0.0000"
                    parts = line.split(',')
                    epoch_part = parts[0].split(':')[0].replace('Epoch ', '').strip()
                    train_loss_part = parts[0].split(':')[2].strip()
                    val_loss_part = parts[1].split(':')[1].strip()
                    char_acc_part = parts[2].split(':')[1].strip()
                    word_acc_part = parts[3].split(':')[1].strip()
                    
                    epochs.append(int(epoch_part))
                    train_losses.append(float(train_loss_part))
                    val_losses.append(float(val_loss_part))
                    char_accs.append(float(char_acc_part))
                    word_accs.append(float(word_acc_part))
                except Exception as e:
                    print(f"Skipping line due to parse error: {line.strip()} | Error: {e}")
                    continue

    if not epochs:
        print("No metrics found in log file.")
        return

    plt.figure(figsize=(12, 5))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, char_accs, label='Char Acc')
    plt.plot(epochs, word_accs, label='Word Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_plot.png')
    print("Plot saved to training_plot.png")
    plt.show()
