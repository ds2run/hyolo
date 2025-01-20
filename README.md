# hYOLO Model: Enhancing Object Classification with Hierarchical Context in YOLOv8

## Project Description
This repository provides the code and resources necessary to reproduce the experiments presented in the paper *"hYOLO Model: Enhancing Object Classification with Hierarchical Context in YOLOv8"*. The project introduces a hierarchical extension to YOLO-based object detection, addressing the limitations of traditional flat classification by capturing object relationships and offering control over the severity of classification errors. Key contributions of this work include:
- A novel hierarchical architecture built upon the YOLO model family.
- A modified loss function for handling hierarchical relationships.
- A performance metric designed specifically for hierarchical classification tasks.

The model is evaluated on two hierarchical categorizations of the same dataset: a **Systematic Categorization**, which organizes objects based on store categories, ignoring visual similarities, and a **Visual Similarity Categorization**, which accounts for shared visual characteristics across classes.

In addition, six different hierarchical architectures were developed and evaluated, demonstrating the model's ability to capture object relationships and outperform traditional flat classification in real-world tasks.

This README provides instructions for using the scripts and the dataset, switching between training and evaluation modes, handling different hierarchical dataset versions, and training and evaluating various hierarchical architectures.

---

## Table of Contents
1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Usage](#usage)
4. [Using Dataset Hierarchy Versions](#using-dataset-hierarchy-versions)
5. [Switching Between Hierarchical Architectures](#switching-between-hierarchical-architectures)
6. [Incorporating a Penalty Term into the Loss Function with Varying Alpha Values](#incorporating-a-penalty-term-into-the-loss-function-with-varying-alpha-values)
7. [Using Hierarchical Evaluation Metrics](#using-hierarchical-evaluation-metrics)
8. [Training with Adjustable Classification Loss Weight](#training-with-adjustable-classification-loss-weight)
9. [Outputs and Directory Structure](#outputs-and-directory-structure)
10. [License](#license)
11. [Acknowledgments](#acknowledgments)
12. [Support](#support)

---

## Installation

1. Clone this repository:
```bash
git clone https://github.com/ds2run/hyolo.git
cd hyolo
```

2. Create and activate a virtual environment (optional, but recommended):
```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Ultralytics:
```bash
pip install -e .
```
---
## Dataset
The 100-class dataset used in the experiments presented in the paper is available upon request via email at <hyolopermission@soft2run.com>. It will be provided to anyone for non-commercial purposes upon receiving a brief explanation of the intended use.

### Steps to download the 100-class Dataset
1. Submit a request for authorization to <hyolopermission@soft2run.com>. Once approved, you will receive the download link.
2. Use the provided link to download the `Grocery_100_Dataset.zip` file.
3. Unarchive `Grocery_100_Dataset.zip`: After downloading, extract the `.zip` file using a standard unarchiving tool:
   - On Linux:
     ```bash
     unzip Grocery_100_Dataset.zip -d Grocery_100_Dataset
     ```
   - On Windows:
     - Right-click on the `.zip` file.
     - Select "Extract All...".
     - Choose the destination folder as `Grocery_100_Dataset/`, or create a new folder with that name.
     - Follow the instructions to complete file extraction.
4. Move the extracted dataset folder `Grocery_100_Dataset` in the `dataset/` subdirectory of `hyolo/` directory.
5. Verify that the following directory structure is present and matches exactly as specified below:

<pre><code class="language-bash">
hyolo/
├── dataset/
    ├── Grocery_100_Dataset/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        ├── labels1/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels2/
            ├── train/
            ├── val/
            └── test/
</code></pre>

| **Directory**               | **Description**                                                                              |
| --------------------------- | -------------------------------------------------------------------------------------------- |
| `hyolo`                     | Root directory of the repository.                                                            |
| `dataset/`                  | Subdirectory containing the datasets used for hierarchical training, validation and testing. |
| `Grocery_100_Dataset/`      | Subdirectory within `dataset/` containing the specific dataset used.                         |


### Inside `Grocery_100_Dataset/` directory
The `Grocery_100_Dataset/` dataset directory consists of three main subdirectories: `images`, `labels1` and `labels2`.
- `images/` : Contains the actual image files (.jpg) divided into three subdirectories:
  - `train/`: Images for training the model.
  - `val/`  : Images for validating the model.
  - `test/` : Images for testing the model.

- `labels1/`: Stores annotation files (.txt) formatted for **hierarchical version 1** of the dataset with three subdirectories: 
  - `train/`: Labels for training images.
  - `val/`: Labels for validation images.
  - `test/`: Labels for testing images.

- `labels2/`: Stores annotation files (.txt) formatted for **hierarchical version 2** of the dataset with three subdirectories:
  - `train/`: Labels for training images.
  - `val/`: Labels for validation images.
  - `test/`: Labels for testing images.


Each .txt file in the `labels1/` directory is paired with a file of the same name in the `images/` directory and corresponds to hierarchical version 1 of the dataset. Subsequently, the directory structure of `labels1/` and `labels2/` mirrors the structure of the `images/` directory. 

### Update your configuration parameters:

1. Set the `dataset_hier_version` parameter to 1 for dataset hierarchy version 1 or 2 for dataset hierarchy version 2
2. Set the `labels_dir_name` parameter in the command line or script to match the label directory you want to use (`labels1` for dataset hierarchy version 1 or `labels2` for dataset hierarchy version 2).
3. Ensure the `hier_depth` parameter matches the corresponding hierarchical version of the dataset( 4 for dataset hierarchy version 1 and 5 for dataset hierarchy version 2).

---
## Usage

### Model Training

To train a hYOLO model from a pretrained *.pt model using a custom hierarchical dataset use:
<pre><code class="language-bash">yolo <span style="color:#d26d6d">detect train</span> \
data=hierarchical_configs/yolov8_100class_hier_v1.yaml model=yolov8n.pt \
epochs=300 patience=50 imgsz=1280 device=cpu dataset_hier_version=1 hier_depth=4 labels_dir_name=labels1 \
dependency_loss=False hier_arch_version=4</code></pre>

Parameters:

| **Parameter**         | **Description**                                                              |
| --------------------- | ---------------------------------------------------------------------------- |
| `data`                | Path to the dataset configuration file (e.g., yolov8_100class_hier_v1.yaml). |
| `model`               | Path to pretrained model weights (.pt file).                                 |
| `epochs`              | Number of training epochs.                                                   |
| `patience`            | Early stopping patience (in epochs).                                         |
| `imgsz`               | Input image size.                                                            |
| `device`              | Hardware for training (CPU or GPU index, e.g., 0).                           |
| `dataset_hier_version`| Hierarchical dataset version to be used for training or evaluation.          |
| `hier_depth`          | Depth of the hierarchy (4 for Dataset V1).                                   |
| `labels_dir_name`     | Name of the directory containing the label files.                            |
| `hier_arch_version`   | Hierarchical architecture version to be used for training or evaluation.     |
| `dependency_loss`     | Whether to include dependency loss during training.                          |

For a full list of parameters, see the `default.yaml` file in [ultralytics/yolo/cfg/default.yaml](ultralytics/yolo/cfg/default.yaml)

The model is automatically evaluated on the `val` dataset split upon training completion.

### Model Evaluation

To evaluate a trained hYOLO model on a custom hierarchical test dataset use:
<pre><code class="language-bash">yolo <span style="color:#d26d6d">detect val split=test</span> \
data=hierarchical_configs/yolov8_100class_hier_v1.yaml dataset_hier_version=1 \
model=runs/hier_yolo/train/weights/best.pt batch=2 hier_depth=4 labels_dir_name=labels1</code></pre>

In this command, the `model` refers to the pretrained weights that were created when the model was trained. These weights are stored in the `weights` directory within the training output folder (e.g., `runs/hier_yolo/train/weights/`). Additionally, the `best.pt` file represents the weights of the model that achieved the best metrics during training. If you want to evaluate a specific model on this dataset, replace `best.pt` with the specific pre-trained weights of your model and provide a correct path to these weights.

For more detailed information about the outputs of the model training process, including the structure of directories and the content of the saved files, please refer to [8. Outputs and Directory Structure](#outputs-and-directory-structure).

---
## Using Dataset Hierarchy Versions
The repository supports two versions of hierarchical categorizations for the dataset:

* Hierarchy Version 1: Logical groupings without visual similarities

	For both training and evaluation, set `dataset_hier_version=1`, `hier_depth=4` and `labels_dir_name=labels1` with `yolov8_100class_hier_v1.yaml`.
	    <pre><code>
	    Parameters:
	        data=hierarchical_configs/yolov8_100class_hier_<span style="color:#d26d6d">v1</span>.yaml
          dataset_hier_version=<span style="color:#d26d6d">1</span>
	        hier_depth=<span style="color:#d26d6d">4</span>
			    labels_dir_name=<span style="color:#d26d6d">labels1</span>
	    </code></pre>

* Hierarchy Version 2: Based on visual similarities across classes

	For both training and evaluation, set `dataset_hier_version=2`, `hier_depth=5` and `labels_dir_name=labels2` with `yolov8_100class_hier_v2.yaml`.
	    <pre><code>
	    Parameters:
	        data=hierarchical_configs/yolov8_100class_hier_<span style="color:#d26d6d">v2</span>.yaml
          dataset_hier_version=<span style="color:#d26d6d">2</span>
          hier_depth=<span style="color:#d26d6d">5</span>
			    labels_dir_name=<span style="color:#d26d6d">labels2</span>
	    </code></pre>

---
## Switching Between Hierarchical Architectures

To train using one of six hierarchical architectures, set the `hier_arch_version` parameter between 1 and 6.

Example:
<pre><code class="language-bash">yolo detect train data=hierarchical_configs/yolov8_100class_hier_v1.yaml \
model=yolov8n.pt epochs=300 patience=50 imgsz=1280 dataset_hier_version=1 hier_depth = 4 \
labels_dir_name=labels1 <span style="color: #d26d6d">hier_arch_version=4</span>
</code></pre>

---
## Incorporating a Penalty Term into the Loss Function with Varying Alpha Values
To add a penalty term to the BCE loss:
* Set `dependency_loss=True`.
* Adjust the penalty value using `regul_alpha`.

Example:
<pre><code class="language-bash">yolo detect train data=hierarchical_configs/yolov8_100class_hier_v2.yaml \
model=yolov8n.pt epochs=300 patience=50 imgsz=1280 dataset_hier_version=2 labels_dir_name=labels2 \
hier_depth=5 hier_arch_version=4 <span style="color: #d26d6d">dependency_loss=True regul_alpha=50</span>
</code></pre>

---
## Using Hierarchical Evaluation Metrics

To use hierarchical F1, Precision, and Recall metrics, set `calc_set_metric=True`, `calc_TP_FN_FP=True`, `get_hier_paths=True`, `calc_TP_FP_conf=True`

Example:
<pre><code class="language-bash">yolo detect val split=test \
data=hierarchical_configs/yolov8_100class_hier_v1.yaml \
model=yolov8n.pt imgsz=1280 dataset_hier_version=1 hier_depth=4 \
labels_dir_name=labels1 hier_arch_version=4 \ 
<span style="color: #d26d6d">calc_set_metric=True calc_TP_FN_FP=True get_hier_paths=True calc_TP_FP_conf=True</span>
</code></pre>

---
## Training with Adjustable Classification Loss Weight

To adjust the weight of BCE loss in the total loss, modify the `cls` parameter. Increasing it from 0.5 to 2.0 notably improved the F1 score.

Example:
<pre><code class="language-bash">yolo detect train data=hierarchical_configs/yolov8_100class_hier_v1.yaml \
model=yolov8n.pt epochs=300 patience=50 imgsz=1280 dataset_hier_version=1 labels_dir_name=labels1 \
hier_depth=4 hier_arch_version=4 <span style="color: #d26d6d">cls=2</span>
</code></pre>

---
## Outputs and Directory Structure

Specify the output directory using the `project` parameter. If the directory does not exist, it will be created with subdirectories for training (train) and validation (val). Subsequent trainings increment the directory name unless explicitly renamed.

Example:
<pre><code class="language-bash">yolo detect train data=hierarchical_configs/yolov8_100class_hier_v2.yaml \
dataset_hier_version=2 labels_dir_name=labels2 hier_depth=5 hier_arch_version=4 \
model=yolov8n.pt <span style="color: #d26d6d">project=runs/hier_yolo</span>
</code></pre>

Training Outputs: train subdirectory contains:
- model weights, logs, metrics, labels, graphs, batch images, etc. relevant to training.
- `results.csv`: Metrics for each epoch.
- `eval_metrics_prec_recall_mAP.csv`: Evaluation results for each hierarchical level and class.
- subdirectory `weights`: This subdirectory contains the weights for the trained models:
    - `last.pt`: Contains the weights from the last epoch of training.
    - `best.pt`: Contains the weights of the model with the best fitness score.
    - Epoch-based weights (`epochxx.pt`): These files correspond to predetermined epochs, where xx represents the epoch number. These weights are saved at specific checkpoints during training (e.g., `epoch10.pt`, `epoch20.pt`, etc.).

Evaluation Outputs: val directory contains:
- `eval_metrics_prec_recall_mAP.csv` with precision, recall, and mAP scores.
- confusion matrix for each hierarchical level
- graphs with metrics, batch images, etc.

---
## License
This project inherits the [AGPL 3.0 license](LICENSE) from YOLOv8.

---
## Acknowledgments
Built on the YOLOv8 framework by Ultralytics release 8.0.58.
[YOLOv8: State-of-the-Art Computer Vision Model](https://yolov8.com/)
