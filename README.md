# GeoCamo

Training code for the geometry guided vehicle camouflage model. This repository contains the surface field, differentiable rendering and training loss, together with the detector modules needed to load a pretrained YOLOv3 checkpoint. It contains no evaluation or test scripts, pretrained weights, meshes, or datasets.

## Environment

The recorded training stack used Python 3.10.8, PyTorch 2.1.2 with CUDA 11.8, torchvision 0.16.2, PyTorch3D 0.7.9, and nvdiffrast 0.4.0. Other Python dependencies and their versions are in `environment.yml` and `requirements.txt`. A CUDA capable GPU and a matching CUDA toolkit (`nvcc`) are needed to build the rendering dependencies.

```bash
conda env create -f environment.yml
conda activate geocamo
python -m pip install 'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9'
python -m pip install 'git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0'
```

The recorded machine had an NVIDIA RTX 4080 SUPER. The exact OpenCV build recorded there was 4.13.0.92; the environment file selects 4.8.0.76 because it is compatible with the recorded NumPy 1.26.4 and the APIs used here.

## Required inputs

* A pretrained YOLOv3 checkpoint readable by `torch.load`, with a `model` entry whose YAML describes the detector. No checkpoint is distributed here.
* One or more vehicle OBJ meshes, with any referenced MTL and texture files. Pass meshes in the same order as their optional face lists.
* A directory of training `.npz` files. Each file contains `img` (a rendered background image) and `cam_trans` (camera transform used by the renderer). By default `img` is interpreted as BGR; set `--npz-color-order rgb` for RGB files.
* Optional YOLO-format `.txt` labels and `.png` vehicle masks with stems matching the NPZ files. If omitted, training derives a vehicle box and mask from its rendered mesh.
* Optional face-list text files containing OBJ face-line indices for the paintable vehicle surfaces. If omitted, the whole mesh is optimized.

## Train

```bash
python train.py \
  --weights /path/to/yolov3.pt \
  --data data/train.yaml \
  --npz-dir /path/to/training/npz \
  --obj-file /path/to/car_1.obj /path/to/car_2.obj \
  --faces /path/to/car_1_faces.txt /path/to/car_2_faces.txt \
  --epochs 5 \
  --block-resolution 144 \
  --subcolor-temp 0.45 \
  --local-color-dist-resolution 12 \
  --lambda-cross 0.02 \
  --lambda-local-color-dist 0.1 \
  --output-dir results/geocamo
```

For one vehicle, provide one OBJ and one face list. `--faces` may be omitted to optimize every mesh face. `--label-dir` and `--mask-dir` may be supplied when the corresponding training annotations are available. `python train.py --help` lists all options.

Training writes epoch and final checkpoints, configuration metadata, loss history, and preview images beneath `--output-dir`. Checkpoint selection by validation is outside this training-only release; the final checkpoint is the result of the complete configured training run.

## Code provenance

The `models/` and `utils/` modules contain the YOLOv3 detector definitions and supporting code needed to load the detector checkpoint, adapted from the [Ultralytics YOLOv3 project](https://github.com/ultralytics/yolov3). The `atf/` package and `train.py` contain the GeoCamo training implementation.
