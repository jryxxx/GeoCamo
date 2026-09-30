# GeoCamo

Code for the paper **GeoCamo: Geometry-Conditioned Texture Fields for Transferable Adversarial Camouflage Against Vehicle Detectors**.

## Abstract

Adversarial camouflage can suppress vehicle detection across viewpoints, but existing textures are often tied to the mesh or projection used for optimization. We propose GeoCamo, a geometry-conditioned texture field for cross-vehicle transfer. Its position and geometry hash branches combine body-relative and local-shape cues, while a functional consistency loss aligns blending and palette decisions across vehicles. The field is optimized on multiple source meshes through differentiable rendering against a fixed detector, with a region-level palette constraint to retain color diversity. A voxel query-and-broadcast operation produces coherent camouflage regions instead of independently colored surface samples. The same frozen field is evaluated on held-out and image-reconstructed meshes without shared texture coordinates, vertex correspondence, or target-specific optimization. On held-out vehicle meshes, it yields lower mean average precision for vehicle detection than the texture-projection baselines. Evaluations across camera configurations and detector architectures further test its transfer beyond the training setup. Tests on reconstructed vehicles and photographed miniature models examine changes in surface representation and image capture. Surface-wise blending and cross-vehicle output analyses show how related body regions receive compatible texture decisions. The results support geometry-conditioned texture generation as a means of transferring adversarial camouflage while maintaining detector suppression. Code is available at https://github.com/jryxxx/GeoCamo.

## Usage

### Environment

The reference environment uses Python 3.10.8, PyTorch 2.1.2 with CUDA 11.8, torchvision 0.16.2, PyTorch3D 0.7.9, and nvdiffrast 0.4.0. Other package versions are pinned in `environment.yml`. A CUDA-capable GPU and a matching CUDA toolkit (`nvcc`) are needed to build the rendering dependencies.

```bash
conda env create -f environment.yml
conda activate geocamo
python -m pip install 'git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9'
python -m pip install 'git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0'
```

The included YOLOv3 detector modules are adapted from [Ultralytics YOLOv3](https://github.com/ultralytics/yolov3).

### Inputs

- A pretrained YOLOv3 checkpoint readable by `torch.load`, with a `model` entry containing the detector configuration.
- Vehicle OBJ meshes with any referenced MTL and texture files, aligned to a consistent canonical orientation.
- A directory of `.npz` files, each containing `img` (background image) and `cam_trans` (camera transform). Images are interpreted as BGR by default; set `--npz-color-order rgb` for RGB files.
- Optionally, one paintable-face list per mesh, in the same order as `--obj-file`. Without `--faces`, every mesh face is optimized. Matching YOLO-format labels and vehicle-mask images may also be supplied with `--label-dir` and `--mask-dir`.

### Run

The paper's main configuration uses five source vehicles. Replace the example paths with your detector checkpoint, input files, and paintable-face lists:

```bash
python train.py \
  --weights /path/to/yolov3.pt \
  --data data/train.yaml \
  --npz-dir /path/to/background_npz \
  --obj-file /path/to/compact.obj /path/to/etron.obj /path/to/minivan.obj /path/to/pickup.obj /path/to/suv.obj \
  --faces /path/to/compact_faces.txt /path/to/etron_faces.txt /path/to/minivan_faces.txt /path/to/pickup_faces.txt /path/to/suv_faces.txt \
  --epochs 5 \
  --block-resolution 144 \
  --subcolor-temp 0.45 \
  --local-color-dist-resolution 12 \
  --lambda-cross 0.02 \
  --lambda-local-color-dist 0.1 \
  --output-dir results/geocamo
```

The command writes checkpoints, configuration metadata, loss history, and preview images to `--output-dir`. Run `python train.py --help` for the full set of options.
