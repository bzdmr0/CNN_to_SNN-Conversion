import os
import sys
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from models.mobilenetv4_classifier import MobileNetV4ShipClassifier
except ImportError:
    from mobilenetv4_classifier import MobileNetV4ShipClassifier
try:
    from models.spiking_layers import ScaledLIFNeuron,OutputLIFNeuron
except ImportError:
    from spiking_layers import ScaledLIFNeuron,OutputLIFNeuron

from utils.fold_bn import fold_bnact_into_conv_fx
from utils.spike_stats import reset_all_spiking_states


def load_checkpoint_safe(checkpoint_path, device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def set_module_by_name(model, module_name, new_module):
    parts = module_name.split(".")
    parent = model

    for part in parts[:-1]:
        parent = getattr(parent, part)

    setattr(parent, parts[-1], new_module)


class SNNMobileNetV4ShipClassifier(nn.Module):
    def __init__(
        self,
        checkpoint_path,
        threshold_path,
        num_classes=2,
        timesteps=128,
        decay=0.99,
        threshold_scale=1.0,
        layer_threshold_scales=None,
        exclude_lif_layers=None,
        scale_output=True,
        use_output_lif=False,
        output_threshold=1.0,
        output_current_mode="shift",
        fold_bn=False,
        zero_classifier_bias=False,
        device=None,
        verbose=True,
    ):
        super().__init__()

        self.checkpoint_path = checkpoint_path
        self.threshold_path = threshold_path
        self.num_classes = num_classes
        self.timesteps = int(timesteps)
        self.decay = float(decay)
        self.threshold_scale = float(threshold_scale)
        self.scale_output = bool(scale_output)
        self.use_output_lif = bool(use_output_lif)
        self.output_threshold = float(output_threshold)
        self.output_current_mode = output_current_mode
        self.verbose = verbose

        self.fold_bn = fold_bn
        self.zero_classifier_bias = zero_classifier_bias

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = MobileNetV4ShipClassifier(
            num_classes=num_classes,
            pretrained=False,
        )

        if exclude_lif_layers is None:
            exclude_lif_layers = []

        self.exclude_lif_layers = set(exclude_lif_layers)

        if layer_threshold_scales is None:
            layer_threshold_scales = {}

        self.layer_threshold_scales = layer_threshold_scales

        checkpoint = load_checkpoint_safe(checkpoint_path, device)

        if "model_state_dict" not in checkpoint:
            raise KeyError("Checkpoint must contain model_state_dict.")

        self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.fold_bn:
            print("\nApplying BN folding...")
            self.model = fold_bnact_into_conv_fx(self.model, verbose=self.verbose)

        if self.zero_classifier_bias:
            self.zero_final_classifier_bias()

        thresholds = torch.load(threshold_path, map_location="cpu")

        if not isinstance(thresholds, dict):
            raise TypeError("threshold_path must contain a dict.")

        self.thresholds = thresholds

        self.replace_relu_with_lif()

        if self.use_output_lif:
            self.output_lif = OutputLIFNeuron(
                threshold=self.output_threshold,
                decay=self.decay,
                threshold_scale=1.0,
                current_mode=self.output_current_mode,
            )
        else:
            self.output_lif = None

    def zero_final_classifier_bias(self):
        """
        Optional bias handling.

        This zeros final classifier bias only.
        Do NOT zero all biases blindly after BN folding.
        """

        if hasattr(self.model, "model") and hasattr(self.model.model, "classifier"):
            classifier = self.model.model.classifier

            if hasattr(classifier, "bias") and classifier.bias is not None:
                with torch.no_grad():
                    classifier.bias.zero_()

                print("Final classifier bias zeroed.")
                return

        print("[WARNING] Final classifier bias not found.")

    def replace_relu_with_lif(self):
        relu_names = []

        for name, module in self.model.named_modules():
            if isinstance(module, nn.ReLU):
                relu_names.append(name)

        if self.verbose:
            print("\nReLU layers found in model:")
            for name in relu_names:
                print(f"  {name}")

            print("\nReplacing ReLU layers with ScaledLIFNeuron:")
            print("==========================================")

        replaced = 0
        missing = 0
        skipped = 0

        for name in relu_names:
            # IMPORTANT: skip excluded layers
            if name in self.exclude_lif_layers:
                skipped += 1
                if self.verbose:
                    print(f"[SKIP] Keeping ReLU unchanged: {name}")
                continue

            if name not in self.thresholds:
                missing += 1
                if self.verbose:
                    print(f"[WARNING] Missing threshold: {name}")
                continue

            threshold = self.thresholds[name]

            layer_scale = self.layer_threshold_scales.get(name, self.threshold_scale)

            lif = ScaledLIFNeuron(
                threshold=threshold,
                decay=self.decay,
                threshold_scale=layer_scale,
                scale_output=self.scale_output,
                name=name,
            )

            set_module_by_name(self.model, name, lif)

            if torch.is_tensor(threshold):
                threshold_info = f"tensor shape={tuple(threshold.shape)}"
            else:
                threshold_info = f"{float(threshold):.6f}"

            if self.verbose:
                print(
                    f"Replaced {name:70s} "
                    f"threshold={threshold_info}, "
                    f"decay={self.decay}, "
                    f"threshold_scale={layer_scale}, "
                    f"scale_output={self.scale_output}"
                )

            replaced += 1

        print("\nReplacement summary:")
        print(f"Replaced ReLU layers: {replaced}")
        print(f"Skipped ReLU layers:  {skipped}")
        print(f"Missing thresholds:   {missing}")

        if replaced == 0:
            raise RuntimeError("No ReLU layers replaced.")

    def reset_states(self):
        reset_all_spiking_states(self.model)

        if self.output_lif is not None:
            self.output_lif.reset_state()

    def forward(self, x):
        self.reset_states()

        logits_sum = None

        if self.output_lif is not None:
            for _ in range(self.timesteps):
                logits_t = self.model(x)
                _ = self.output_lif(logits_t)

            return self.output_lif.spike_count

        for _ in range(self.timesteps):
            logits_t = self.model(x)

            if logits_sum is None:
                logits_sum = logits_t
            else:
                logits_sum = logits_sum + logits_t

        return logits_sum / self.timesteps