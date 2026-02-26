import os.path

import torch
import tqdm
import argparse
from pathlib import Path
from terratorch.tasks.tiled_inference import tiled_inference
from terratorch.cli_tools import LightningInferenceModel, open_tiff, write_tiff

parser = argparse.ArgumentParser()
parser.add_argument("-c" ,'--config', type=str, required=True)
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("-o" ,'--output_dir', type=str, required=True)
parser.add_argument('--predict_split', type=str, default=None)
parser.add_argument("-v" ,'--verbose', action='store_true')
parser.add_argument( '--overwrite', action='store_true')
args = parser.parse_args()

# Alternatively build the model from a config file
task = LightningInferenceModel.from_config(
    args.config,
    args.ckpt,
)

if torch.cuda.is_available():
    device = 'cuda'
elif torch.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

task.model.eval()
task.model.to(device)

task.datamodule.batch_size = 1
task.datamodule.predict_split = args.predict_split
task.datamodule.setup("predict")

data_loader = task.datamodule.predict_dataloader()
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
tif_dir = Path(task.datamodule.predict_data_root) / "DEM"

# Inference wrapper for TerraTorch task model
def model_forward(x, **kwargs):
    # Returns a torch Tensor
    return task.model(x, **kwargs).output

print(f"Saving output to {output_dir}")

for batch in tqdm.tqdm(data_loader):
    filename = batch["filename"][0]
    print(filename)
    out_file_name = output_dir / (os.path.basename(filename).rsplit('_DEM.tif')[0] + "_prediction.tif")
    if not args.overwrite and out_file_name.exists():
        print(f"Skipping {out_file_name} ...")

    # Run tiled inference (data is loaded automatically to GPU)
    input = task.datamodule.aug(batch)["image"]
    pred = tiled_inference(
        model_forward,
        input,
        crop=256,
        stride=208,  # Overlap of 16 pixel on each side, 8 pixels dropped
        batch_size=64,
        delta=8,
        verbose=args.verbose,
        device=device,
    )

    # Remove batch dim and compute segmentation map
    pred = pred.squeeze(0).argmax(dim=0).cpu().numpy()

    # Save image
    print(out_file_name)
    mask, metadata = open_tiff(filename)
    if args.verbose:
        print(f"Saving output to {out_file_name}")
    write_tiff(pred, out_file_name, metadata)

print("Done")
