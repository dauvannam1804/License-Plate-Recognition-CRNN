
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

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
    Reads the CSV log file and plots training metrics.
    Expected format: epoch,train_loss,train_char_acc,train_word_acc,val_loss,val_char_acc,val_word_acc
    """
    epochs = []
    train_losses = []
    val_losses = []
    train_char_accs = []
    val_char_accs = []
    train_word_accs = []
    val_word_accs = []
    
    if not os.path.exists(log_file):
        print(f"Log file {log_file} not found.")
        return

    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epochs.append(int(row['epoch']))
                train_losses.append(float(row['train_loss']))
                val_losses.append(float(row['val_loss']))
                train_char_accs.append(float(row['train_char_acc']))
                val_char_accs.append(float(row['val_char_acc']))
                train_word_accs.append(float(row['train_word_acc']))
                val_word_accs.append(float(row['val_word_acc']))
            except Exception as e:
                print(f"Skipping row due to parse error: {row} | Error: {e}")
                continue

    if not epochs:
        print("No metrics found in log file.")
        return

    plt.figure(figsize=(15, 5))
    
    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Val Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Word Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_word_accs, label='Train Word Acc', marker='o')
    plt.plot(epochs, val_word_accs, label='Val Word Acc', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Word Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot Char Accuracy
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_char_accs, label='Train Char Acc', marker='o')
    plt.plot(epochs, val_char_accs, label='Val Char Acc', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Char Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    output_png = log_file.replace('.csv', '.png')
    plt.savefig(output_png)
    print(f"Plot saved to {output_png}")
    plt.show()
