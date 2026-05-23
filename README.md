# CNN to SNN Conversion for Ship Classification

This project converts a MobileNetV4-based convolutional neural network into a Leaky Integrate-and-Fire Spiking Neural Network for binary satellite image classification.

The target task is to classify each image as either `no_ship` or `ship`. The final package also includes hardware-oriented SNN artifacts for FPGA/ASIC-style implementation.

## Highlights

| Item | Value |
| --- | --- |
| Task | Binary satellite image classification |
| Classes | `no_ship`, `ship` |
| ANN backbone | MobileNetV4-Conv-S |
| Input size | `224 x 224 x 3` |
| SNN neuron | Leaky Integrate-and-Fire |
| Conversion method | ANN-to-SNN conversion |
| Threshold type | Channel-wise activation thresholds |
| Final readout | Output LIF spike count |
| Recommended timestep | `T = 128` |
| Hardware focus | Folded Conv/BN, LIF thresholds, exported weights |

The best SNN setup uses the ANN weights trained on the augmented 80k dataset and channel-wise thresholds calibrated on the 30k dataset. This cross-calibrated setup produced the strongest SNN results in the experiments.

## Repository Contents

```text
.
|-- CNN_to_SNN_Summary.txt
|-- README.md
`-- SNN_Hardware_Implementation_Package/
    |-- best_config.txt
    |-- checkpoints/
    |   |-- aug_80k_best_mobilenetv4_ship_classifier.pth
    |   `-- aug_80k_on30k_channelwise_thresholds.pth
    |-- hardware_export/
    |   |-- hardware_manifest.json
    |   |-- *_weight.pt
    |   |-- *_bias.pt
    |   `-- *_threshold.pt
    |-- models/
    |   |-- mobilenetv4_classifier.py
    |   |-- snn_mobilenetv4_classifier.py
    |   `-- spiking_layers.py
    |-- snn_hardware_layer_summary.csv
    `-- utils/
        |-- fold_bn.py
        `-- spike_stats.py
```

## Model Architecture

The ANN classifier is built with `timm`:

```python
model_name = "hf_hub:timm/mobilenetv4_conv_small.e2400_r224_in1k"
```

The classifier head is adapted to two classes. During conversion, ReLU activations are replaced with calibrated LIF neurons.

Original ANN block:

```text
Conv -> BatchNorm -> ReLU
```

Hardware-friendly SNN block:

```text
FusedConv -> LIF
```

BatchNorm can be folded into the preceding convolution, so the exported hardware graph does not require a separate BatchNorm block.

## Conversion Strategy and References

The ANN-to-SNN conversion strategy used in this project was inspired by prior CNN-to-SNN conversion methods, including rate-based spiking inference, threshold calibration, spike-count output decisions, and BN/activation normalization techniques. The final model follows a LIF-based conversion approach where ReLU activations are replaced by calibrated LIF neurons, convolutional weights are preserved, BatchNorm is folded into Conv layers, and the output decision is made using spike counts over multiple timesteps.

References used for ANN-to-SNN conversion:

1. MathWorks, "Convert Convolutional Network to Spiking Neural Network"  
   Used for: timestep-based SNN inference, spike-count classification, CNN-to-SNN workflow.

2. Dengyu Wu, SpKeras CNN-to-SNN Conversion Toolbox  
   Used for: CNN pretraining, CNN-to-SNN conversion flow, extraction of weights/bias/thresholds.

3. IEEE / paper on CNN-to-SNN conversion  
   Used for: LIF neuron behavior, reset-by-subtraction, BN folding, threshold normalization, timestep effect.

## LIF Neuron Dynamics

The hidden SNN layers use `ScaledLIFNeuron`:

```text
x_scaled = x / effective_threshold
mem = mem * decay + x_scaled
spike = mem >= 1
mem = mem - spike
```

When `scale_output=True`, the spike output is multiplied back by the effective threshold. This helps keep the activation scale closer to the original ANN layers.

The final classifier can also use `OutputLIFNeuron`. In the final configuration, the model compares accumulated output spike counts and selects the class with the larger count.

## Final Configuration

The selected configuration is stored in:

```text
SNN_Hardware_Implementation_Package/best_config.txt
```

```python
checkpoint_path = "checkpoints/aug_80k_best_mobilenetv4_ship_classifier.pth"
threshold_path = "checkpoints/aug_80k_on30k_channelwise_thresholds.pth"

timesteps = 128
decay = 0.99
threshold_scale = 1.0
scale_output = True

use_output_lif = True
output_threshold = 1.0
output_current_mode = "shift"

layer_threshold_scales = {}
```

`T = 128` was selected as the main latency/accuracy trade-off point, while 
`T = 256` and `T = 192` give slightly higher accuracy at higher inference cost.

## Results

### ANN Baselines

| Model | Accuracy | no_ship Recall | ship Recall |
| --- | ---: | ---: | ---: |
| 30k ANN | 0.9556 | 0.9471 | 0.9640 |
| 80k ANN | 0.9588 | 0.9457 | 0.9715 |

The 80k ANN was selected as the final ANN baseline because it slightly improved total accuracy and improved ship recall.

### Final Channel-Wise SNN

| Model | Checkpoint | Thresholds | Readout | T | Accuracy | Balanced Accuracy |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Final SNN | 80k | 30k channel-wise | Output LIF spike count | 128 | 0.9500 | 0.9493 |
| Final SNN | 80k | 30k channel-wise | Output LIF spike count | 192 | 0.9550 | 0.9520 |
| Final SNN | 80k | 30k channel-wise | Output LIF spike count | 256 | 0.9575 | 0.9540 |

Channel-wise threshold calibration was important. It improved SNN stability by assigning a separate threshold to each output channel instead of using a single scalar threshold per layer.

## Development Pipeline

The full project workflow used to produce the final package was:

1. Prepare dataset splits.

```bash
python split_dataset.py
python split_30k.py
python split_extra_dataset.py
```

2. Train ANN classifiers.

```bash
python train_classifier.py
python train30k.py
python train80k.py
```

or run the follow-up training sequence:

```powershell
.\run_followup_training.ps1
```

3. Calibrate SNN thresholds.

```bash
python calibrate_thresholds.py
python calibrate_channelwise_thresholds.py
```

The final model uses:

```text
80k ANN checkpoint + 30k channel-wise threshold calibration
```

4. Evaluate SNN variants and timestep settings.

```bash
python evaluations/evaluate_snn_classifier.py
python evaluations/evaluate_snn_30k_80k_timestep_compare.py
python evaluations/evaluate_output_lif_sweep.py
python grid_search_snn.py
python eval_all_stage_50.py
```

5. Analyze spike activity and export hardware artifacts.

```bash
python spike_activity_analyzer.py
python export_snn_layer_summary.py
python export_hardware_artifacts.py
```

## Quick Start

Install the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Run from inside the package directory:

```bash
cd SNN_Hardware_Implementation_Package
```

Example inference with the final SNN:

```python
import torch
from models.snn_mobilenetv4_classifier import SNNMobileNetV4ShipClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SNNMobileNetV4ShipClassifier(
    checkpoint_path="checkpoints/aug_80k_best_mobilenetv4_ship_classifier.pth",
    threshold_path="checkpoints/aug_80k_on30k_channelwise_thresholds.pth",
    num_classes=2,
    timesteps=128,
    decay=0.99,
    threshold_scale=1.0,
    scale_output=True,
    use_output_lif=True,
    output_threshold=1.0,
    output_current_mode="shift",
    fold_bn=True,
    zero_classifier_bias=True,
    device=device,
)

model = model.to(device)
model.eval()

x = torch.randn(1, 3, 224, 224).to(device)

with torch.no_grad():
    spike_counts = model(x)
    prediction = spike_counts.argmax(dim=1)

print("Spike counts:", spike_counts)
print("Prediction:", prediction.item())
```

Class mapping:

```text
0 -> no_ship
1 -> ship
```

## Hardware Export

The `hardware_export/` directory contains the tensors and manifest needed for a hardware implementation.

Important files:

| File | Purpose |
| --- | --- |
| `hardware_manifest.json` | Ordered layer graph, tensor paths, convolution metadata, LIF settings |
| `*_weight.pt` | Exported Conv2d/Linear weights |
| `*_bias.pt` | Exported bias tensors |
| `*_threshold.pt` | Exported LIF threshold tensors |
| `snn_hardware_layer_summary.csv` | Conv type, channel count, threshold shape, spike-rate summary |

Hardware implementation notes:

- Depthwise and pointwise convolutions should be implemented separately.
- LIF state memory is required for each activation tensor.
- Channel-wise thresholds require per-channel threshold storage.
- BatchNorm is folded into convolution weights and biases.
- The output decision is based on spike-count `argmax`.
- The reset method is subtraction reset.

## Threshold Format

The final SNN uses channel-wise thresholds:

| Threshold type | Shape | Meaning |
| --- | --- | --- |
| Layer-wise | scalar | One threshold for the whole layer |
| Channel-wise | `[1, C, 1, 1]` | One threshold per output channel |

This is especially useful for deeper MobileNetV4 layers, where activation ranges vary strongly across channels.

## Notes

- The GitHub-facing package is focused on the converted SNN and hardware implementation artifacts.
- Raw datasets are not required for loading the packaged SNN, but they are required to reproduce training, calibration, and evaluation.
- Some script defaults were used during experimentation and may need path updates if the project is reorganized.
- The `timm` model name uses Hugging Face Hub. Loading pretrained weights may require internet access, but the packaged SNN loads from local checkpoints with `pretrained=False`.

## License

No license file is included yet. Add a license before public reuse or redistribution.
