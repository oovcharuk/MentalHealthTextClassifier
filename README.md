# MentalHealthTextClassifier

---

## Datasets

This project utilizes several publicly accessible datasets, all of which are stored in [`datasets/`](datasets/):

| Purpose               | Dataset / Link                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------- |
| PTSD                  | [Human Stress Prediction](https://www.kaggle.com/datasets/kreeshrajani/human-stress-prediction)                     |
| PTSD                  | [aya_ptsd](https://www.kaggle.com/datasets/abdelrahmanahmed3/aya-ptsd)                                              |
| Other Disorders       | [Text Classification](https://www.kaggle.com/datasets/comsys/text-classification/data)                                                      |
| Healthy               | [Depression: Reddit Dataset (Cleaned)](https://www.kaggle.com/datasets/infamouscoder/depression-reddit-cleaned)     |

We sincerely appreciate the efforts of all authors and organisations who provided these datasets.

---

## Installation

1. **Create a virtual environment**

   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

2. **Install PyTorch**
   *Pick the command that matches your CUDA version (or CPU-only).*
   Example for CUDA 12.1:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

   For CPU-only:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Install project dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---