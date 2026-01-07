# License Plate Recognition - CRNN

This repository lists a simple CRNN methodology (CNN + LSTM + CTC) to recognize text on European license plates.

## 1. Dataset Preparation

The model expects the data to be in a specific structure.

1.  **Download**: Download the **European License Plates Dataset** from Kaggle:
    [https://www.kaggle.com/datasets/abdelhamidzakaria/european-license-plates-dataset](https://www.kaggle.com/datasets/abdelhamidzakaria/european-license-plates-dataset)

2.  **Unzip**:
    Unzip the downloaded archive. The code expects the folder `dataset_final` to be at the root level (or parallel to `simple_crnn`).
    
    Assuming you have `archive.zip`, run:
    ```bash
    unzip archive.zip -d .
    # This should create a folder named 'dataset_final' containing 'train', 'val', 'test' subfolders.
    ```
    
    **Structure**:
    ```
    .
    ├── dataset_final
    │   ├── train
    │   ├── val
    │   └── test
    ├── simple_crnn
    │   ├── train.py
    │   ├── dataset.py
    │   ├── ...
    └── README.md
    ```

3.  **EDA (Optional)**:
    You can run `eda_dataset_final.ipynb` to explore class distributions and visualize samples.

## 2. Training

The training code is located in the `simple_crnn` directory.

```bash
cd simple_crnn
python3 train.py
```

-   **Configuration**: You can edit `train.py` to change `BATCH_SIZE`, `EPOCHS`, or `LEARNING_RATE`.
-   **Checkpoints**: The best model (highest Word Accuracy) will be saved to `simple_crnn/checkpoints/best_model.pth`.
-   **Metrics**: The script logs CTC Loss, Character Accuracy (positional match), and Word Accuracy (full match).

## 3. Inference

To recognize text on a single image:

```bash
cd simple_crnn
python3 inference.py --image ../dataset_final/test/some_image.png --model checkpoints/best_model.pth
```

-   `--image`: Path to input image.
-   `--model`: Path to trained `.pth` file (defaults to `checkpoints/best_model.pth`).
-   `--dataset`: Path to dataset root (used to rebuild vocabulary from training set). Defaults to `../dataset_final`.

## 4. Implementation Details

-   **Model**: VGG-style CNN backbone + 2-layer Bidirectional LSTM + Linear Classification.
-   **Input**: Images are resized to `32x128`.
-   **Loss**: CTC Loss.
-   **Accuracy**:
    -   **Char**: Percentage of characters matching at the exact position `pred[i] == target[i]`.
    -   **Word**: Percentage of full strings matching exactly.