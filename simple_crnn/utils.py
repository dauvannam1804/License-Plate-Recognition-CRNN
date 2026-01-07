
import torch
import numpy as np

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
