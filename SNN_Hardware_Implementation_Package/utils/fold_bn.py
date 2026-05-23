import copy
import torch
import torch.nn as nn
import torch.fx as fx


class ActOnlyWrapper(nn.Module):
    """
    Replaces BatchNormAct2d after BN folding.

    Important:
    It keeps the activation name as '.act', so old threshold names like
    model.bn1.act
    still exist.
    """

    def __init__(self, act):
        super().__init__()
        self.act = act

    def forward(self, x):
        return self.act(x)


def get_module_by_name(model, name):
    current = model

    for part in name.split("."):
        current = getattr(current, part)

    return current


def set_module_by_name(model, name, new_module):
    parts = name.split(".")
    parent = model

    for part in parts[:-1]:
        parent = getattr(parent, part)

    setattr(parent, parts[-1], new_module)


def is_bn_like(module):
    return (
        hasattr(module, "running_mean")
        and hasattr(module, "running_var")
        and hasattr(module, "weight")
        and hasattr(module, "bias")
        and hasattr(module, "eps")
    )


def fuse_conv_bn_weights(conv, bn):
    """
    Fold BatchNorm parameters into Conv2d.
    """

    if not isinstance(conv, nn.Conv2d):
        raise TypeError("conv must be nn.Conv2d")

    w_conv = conv.weight.detach().clone()

    if conv.bias is None:
        b_conv = torch.zeros(conv.out_channels, device=w_conv.device)
    else:
        b_conv = conv.bias.detach().clone()

    gamma = bn.weight.detach().clone()
    beta = bn.bias.detach().clone()
    mean = bn.running_mean.detach().clone()
    var = bn.running_var.detach().clone()
    eps = bn.eps

    scale = gamma / torch.sqrt(var + eps)

    w_fused = w_conv * scale.reshape(-1, 1, 1, 1)
    b_fused = beta + (b_conv - mean) * scale

    fused_conv = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
        padding_mode=conv.padding_mode,
    )

    fused_conv._bn_folded = True
    fused_conv._folded_from_bn = None

    fused_conv.weight.data.copy_(w_fused)
    fused_conv.bias.data.copy_(b_fused)

    return fused_conv


def get_activation_from_bnact(bn):
    """
    timm BatchNormAct2d usually has bn.act.
    We keep that activation after folding.
    """

    if hasattr(bn, "act"):
        return copy.deepcopy(bn.act)

    return nn.Identity()


def fold_bnact_into_conv_fx(model, verbose=True):
    """
    Finds Conv2d -> BatchNorm-like modules using torch.fx.
    Folds BN into Conv.
    Replaces BNAct module with ActOnlyWrapper(act).

    This preserves names like:
        model.bn1.act
        model.blocks.3.5.dw_mid.bn.act
    """

    model.eval()

    traced = fx.symbolic_trace(model)

    folded = 0
    skipped = 0

    modules = dict(model.named_modules())

    for node in traced.graph.nodes:
        if node.op != "call_module":
            continue

        bn_name = node.target

        if bn_name not in modules:
            continue

        bn_module = modules[bn_name]

        if not is_bn_like(bn_module):
            continue

        if len(node.args) == 0:
            continue

        input_node = node.args[0]

        if not isinstance(input_node, fx.Node):
            continue

        if input_node.op != "call_module":
            continue

        conv_name = input_node.target

        if conv_name not in modules:
            continue

        conv_module = modules[conv_name]

        if not isinstance(conv_module, nn.Conv2d):
            continue

        # Safety: conv output should only feed this BN node
        if len(input_node.users) != 1:
            skipped += 1
            if verbose:
                print(f"[SKIP] Conv has multiple users: {conv_name}")
            continue

        fused_conv = fuse_conv_bn_weights(conv_module, bn_module)
        fused_conv._folded_from_bn = bn_name
        act = get_activation_from_bnact(bn_module)

        set_module_by_name(model, conv_name, fused_conv)
        set_module_by_name(model, bn_name, ActOnlyWrapper(act))

        folded += 1

        if verbose:
            print(f"[FOLDED] {conv_name} + {bn_name}")

    print("\nBN Folding Summary")
    print("==================")
    print(f"Folded Conv+BN pairs: {folded}")
    print(f"Skipped pairs:        {skipped}")

    if folded == 0:
        print("[WARNING] No Conv+BN pairs were folded. Check torch.fx tracing.")

    return model