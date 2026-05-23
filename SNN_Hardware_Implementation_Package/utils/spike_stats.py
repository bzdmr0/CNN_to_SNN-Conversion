from models.spiking_layers import ScaledLIFNeuron, OutputLIFNeuron


def reset_all_spiking_states(model):
    for module in model.modules():
        if hasattr(module, "reset_state"):
            module.reset_state()


def reset_all_spike_stats(model):
    for module in model.modules():
        if hasattr(module, "reset_stats"):
            module.reset_stats()


def collect_spike_rates(model):
    rates = {}

    for name, module in model.named_modules():
        if isinstance(module, ScaledLIFNeuron):
            rates[name] = module.get_spike_rate()

    return rates


def print_spike_rates(model, max_items=None):
    rates = collect_spike_rates(model)

    print("\nSpike Rates")
    print("===========")

    items = list(rates.items())

    if max_items is not None:
        items = items[:max_items]

    for name, rate in items:
        print(f"{name:80s}: {rate:.6f}")

    if len(rates) == 0:
        print("No ScaledLIFNeuron layers found.")