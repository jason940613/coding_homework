# ML1141 – YOLO12n Training and Evaluation

This repository is a machine learning course assignment for training and evaluating a YOLO-based object detection model.

The project focuses on training a YOLO12n model and analyzing its detection performance using a custom evaluation script.

---

## Project Structure


> Note: Datasets, model weights, and training outputs are not included in this repository.

---

## Workflow

1. **Model Training**  
   The YOLO12n model is trained using `train.ipynb` with the prepared training images and labels.

2. **Inference**  
   After training, the best-performing model checkpoint (`best.pt`) is used to generate prediction labels on the test dataset.

3. **Evaluation**  
   The prediction results are evaluated using `eval_yolo_custom.py`, which compares model outputs with ground truth labels and computes:
   - True Positive (TP)
   - False Positive (FP)
   - False Negative (FN)

---

## Training Configuration

- Model: YOLO12n  
- Epochs: 10  
- Image size: 640  
- Batch size: 16  
- Python version: 3.10  

---

## Evaluation Method

The evaluation script performs IoU-based matching between ground truth and predicted bounding boxes:

- IoU threshold: 0.5  
- A prediction is considered **TP** if the class matches and IoU ≥ threshold  
- Unmatched predictions are counted as **FP**  
- Unmatched ground truth boxes are counted as **FN**

The script also generates visualization results to assist in error analysis.

---

## Notes

- This repository contains only source code.
- Large files such as datasets, trained weights (`.pt`), and output results are excluded.
- This project is intended for academic coursework only.

---

## Author

- Course: Machine Learning
- GitHub: https://github.com/jason940613/coding_homework
