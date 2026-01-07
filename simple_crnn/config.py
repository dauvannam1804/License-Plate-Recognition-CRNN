
import os
import torch

# Paths
DATASET_ROOT = '../dataset_final'
CHECKPOINT_DIR = '../checkpoints'
LOG_FILE = '../train.log.csv'

# Data
IMG_HEIGHT = 32
IMG_WIDTH = 128

# Model
HIDDEN_SIZE = 256

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure directories exist
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
