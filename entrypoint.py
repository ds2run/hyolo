import argparse

from ultralytics import YOLO

# Create an argument parser
parser = argparse.ArgumentParser(description='Example Argument Parser')

# Add the model argument with a default value of 'default_model'
parser.add_argument('-m',
                    '--model',
                    type=str,
                    default="yolov8n.pt",
                    help='Specify the model (default: yolov8n.pt)')

# Add the data argument with a default value of 'default_data'
parser.add_argument(
    '-d',
    '--data',
    type=str,
    default="yolov8-6class.yaml",
    help='Specify the input data (default: yolov8-6class.yaml)')

parser.add_argument('-e',
                    '--epochs',
                    type=str,
                    default=1,
                    help='Specify the number of epochs (default: 1)')

parser.add_argument('-b',
                    '--batch',
                    type=str,
                    default=1,
                    help='Specify the batch size (default: 1)')

parser.add_argument('-hier_arch_version',
                    '--hier_arch_version',
                    type=int,
                    default=4,
                    help='Specify the hier_arch_version (default: 4)')

# Parse the command-line arguments
args = parser.parse_args()

# Access the values of the arguments
model_name = args.model
data = args.data
epochs = int(args.epochs)
batch = int(args.batch)
hier_arch_version = int(args.hier_arch_version)

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model from coco

# Train the model
model.train(
    task="detect",
    data="./hierarchical_configs/yolov8_100class_hier_v2.yaml",
    epochs=1,
    batch=5,
    imgsz=320,
    hier_depth=5,
    mosaic=0,
    hier_arch_version=4,
    dependency_loss=False,
    calc_TP_FN_FP=False,
    calc_set_metric=False,
    get_hier_paths=False,
    calc_TP_FP_conf=False,
    device=0)

# Evaluate model on the test split
# metrics = model.val(
#     data="./hierarchical_configs/yolov8_100class_hier_v2.yaml",
#     split="test",  # evaluate on the test set, could be "val"
#     batch=5)  # evaluate model performance on the validation set
