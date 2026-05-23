import torch
import torch.nn as nn


class ScaledLIFNeuron(nn.Module):
    """
    Calibrated Leaky Integrate-and-Fire neuron.

    Dynamics:
        x_scaled = x / effective_threshold
        mem = mem * decay + x_scaled
        spike = mem >= 1
        mem = mem - spike

    If scale_output=True:
        returns spike * effective_threshold

    This keeps the next ANN layers closer to the original activation scale.
    """

    def __init__(
        self,
        threshold=1.0,
        decay=0.99,
        threshold_scale=1.0,
        scale_output=True,
        name=None,
    ):
        super().__init__()

        
        if not torch.is_tensor(threshold):
            threshold = torch.tensor(float(threshold), dtype=torch.float32)
        else:
            threshold = threshold.detach().clone().float()

        self.register_buffer("threshold", threshold)

        self.decay = float(decay)
        self.threshold_scale = float(threshold_scale)
        self.scale_output = bool(scale_output)
        self.name = name

        self.mem = None

        self.total_spikes = 0.0
        self.total_neurons = 0.0
        self.forward_calls = 0

    def reset_state(self):
        self.mem = None

    def reset_stats(self):
        self.total_spikes = 0.0
        self.total_neurons = 0.0
        self.forward_calls = 0

    def get_spike_rate(self):
        if self.total_neurons == 0:
            return 0.0
        return self.total_spikes / self.total_neurons

    def forward(self, x):
        if self.mem is None:
            self.mem = torch.zeros_like(x)

        effective_threshold = self.threshold * self.threshold_scale
        effective_threshold = torch.clamp(effective_threshold, min=1e-6)

        x_scaled = x / effective_threshold

        self.mem = self.mem * self.decay + x_scaled

        spike = (self.mem >= 1.0).float()

        # subtraction reset
        self.mem = self.mem - spike

        self.total_spikes += spike.detach().sum().item()
        self.total_neurons += spike.numel()
        self.forward_calls += 1

        if self.scale_output:
            return spike * effective_threshold

        return spike


class OutputLIFNeuron(nn.Module):
    """
    Output LIF neuron for classification.

    It receives class logits/current and produces output spikes.

    Important:
        Final classifier logits can be negative.
        For output spiking, we convert logits into non-negative current.

    current_mode:
        "shift"  -> x = x - min(x per sample)
        "clamp"  -> x = clamp(x, min=0)
        "none"   -> use raw logits directly
    """

    def __init__(
        self,
        threshold=1.0,
        decay=0.99,
        threshold_scale=1.0,
        current_mode="shift",
    ):
        super().__init__()

        self.threshold = float(threshold)
        self.decay = float(decay)
        self.threshold_scale = float(threshold_scale)
        self.current_mode = current_mode

        self.mem = None
        self.spike_count = None

    def reset_state(self):
        self.mem = None
        self.spike_count = None

    def prepare_current(self, x):
        if self.current_mode == "shift":
            # Preserve relative class difference, make currents non-negative.
            # Shape: [B, num_classes]
            x_min = x.min(dim=1, keepdim=True).values
            x = x - x_min
            return x

        if self.current_mode == "clamp":
            return torch.clamp(x, min=0.0)

        if self.current_mode == "none":
            return x

        raise ValueError(f"Unknown current_mode: {self.current_mode}")

    def forward(self, x):
        if self.mem is None:
            self.mem = torch.zeros_like(x)
            self.spike_count = torch.zeros_like(x)

        effective_threshold = self.threshold * self.threshold_scale

        if effective_threshold < 1e-6:
            effective_threshold = 1.0

        x = self.prepare_current(x)

        self.mem = self.mem * self.decay + x / effective_threshold

        spike = (self.mem >= 1.0).float()

        # subtraction reset
        self.mem = self.mem - spike

        self.spike_count = self.spike_count + spike

        return spike