# Eye Disease Detection

Classifies eye disease images into five categories (Glaucoma, Cataracts, Uveitis, Crossed Eyes, Bulging Eyes) using transfer learning with a fine-tuned ResNet34 model. Includes a separate image augmentation pipeline that generates flipped and rotated copies of the original dataset to increase training data.

## Dataset

Eye disease images organized into five class folders. The original dataset is augmented via horizontal flips and 180-degree rotations (see `augment_images.ipynb`). Both the Original Dataset and Augmented Dataset are included.

## Tech Stack

- fastai / fastcore
- PyTorch (via fastai)
- Pillow
- pandas
- numpy
- matplotlib

## Results

A fine-tuned ResNet34 model is trained for 50 epochs on the augmented dataset to classify five eye disease categories: Glaucoma, Cataracts, Uveitis, Crossed Eyes, and Bulging Eyes. See notebook for detailed results.

## How to Run

### Web app

Install dependencies and start the Flask app:

```bash
python -m pip install -r requirements.txt
python app.py
```

Open <https://localhost:5000>. The development server uses a temporary certificate, so your browser may show a certificate warning. The app requires `eye_disease_model.pkl` in the project root.

### Training

1. Place the dataset in `Augmented Dataset/`, with one subfolder per class.
2. Run `augment_images.ipynb` if you need to generate augmented images from `Original Dataset/`.
3. Open `model.ipynb`, update the dataset `path` for your machine, and run the notebook to train and export the model.
