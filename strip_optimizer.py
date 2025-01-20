from ultralytics.yolo.utils.torch_utils import strip_optimizer

import glob
import sys

from tqdm import tqdm

assert len(sys.argv) == 2

files = glob.glob(sys.argv[1] + "/*.pt")
print(f"Working on: {sys.argv[1]}")

for f in tqdm(files):
	strip_optimizer(f)
