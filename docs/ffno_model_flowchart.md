# FFNO3D Model Flowchart

Source: [`models/ffno_model.py`](../models/ffno_model.py)

## Full Forward Pass

```mermaid
flowchart TD
    X["x: [B, in_channels, D, H, W]"] --> LIFT["lift<br/>Conv3d(in_channels -> width, 1)<br/>GroupNorm<br/>GELU"]
    Z["z_scale: [B, D, H, W] or [B, 1, D, H, W]"] --> BLOCKS
    DX["dx"] --> BLOCKS
    DY["dy"] --> BLOCKS
    LIFT --> H0["hidden: [B, width, D, H, W]"]
    H0 --> BLOCKS["repeat n_layers x<br/>FFNOBlock3dBalanced"]
    BLOCKS --> HEAD1["proj1<br/>Conv3d(width -> 2*width, 1)<br/>GELU"]
    HEAD1 --> HEAD2["proj2<br/>Conv3d(2*width -> out_channels, 1)"]
    HEAD2 --> Y["output: [B, out_channels, D, H, W]"]

    TRAIN{"training and<br/>checkpoint_blocks and<br/>not collect_stats?"} -. wraps lift/block/head .-> LIFT
    TRAIN -. wraps lift/block/head .-> BLOCKS
    TRAIN -. wraps lift/block/head .-> HEAD1
```

## `FFNOBlock3dBalanced`

```mermaid
flowchart TD
    X["x: [B, width, D, H, W]"] --> RES["residual = x"]

    X --> SPEC["SpectralConv2dFull(x, dx, dy)"]
    DX["dx"] --> SPEC
    DY["dy"] --> SPEC
    SPEC --> NS["GroupNorm<br/>GELU<br/>optional spec_dropout"]

    X --> VERT["BalancedVerticalPhysicsStack(x, z_scale)"]
    Z["z_scale"] --> VERT
    VERT --> NV["GroupNorm<br/>GELU<br/>optional vertical_dropout"]

    X --> GATE["branch_gate<br/>Conv3d(3w -> w, 1)<br/>GELU<br/>Conv3d(w -> 2w, 1)<br/>Sigmoid"]
    NS --> GATE
    NV --> GATE
    GATE --> SG["spec_gate"]
    GATE --> VG["vertical_gate"]

    NS --> SMIX["spec_mix = spec * spec_gate"]
    SG --> SMIX
    NV --> VMIX["vert_mix = vert * vertical_gate"]
    VG --> VMIX

    SMIX --> CAT["concat(spec_mix, vert_mix)<br/>[B, 2w, D, H, W]"]
    VMIX --> CAT
    CAT --> FUSE["fuse<br/>Conv3d(2w -> 2w, 1)<br/>GELU<br/>Depthwise Conv3d(2w -> 2w, 3)<br/>GELU<br/>Conv3d(2w -> w, 1)"]
    SMIX --> AVG["0.5 * (spec_mix + vert_mix)"]
    VMIX --> AVG
    FUSE --> FADD["fused + average branch mix"]
    AVG --> FADD
    FADD --> NF["GroupNorm<br/>GELU"]

    RES --> X1["x1 = residual + res_fused * fused"]
    NF --> X1
    X1 --> PW["pw<br/>Conv3d(w -> 2w, 1)<br/>GELU<br/>Conv3d(2w -> w, 1)<br/>GroupNorm"]
    PW --> X2["x2 = x1 + res_pw * pw"]
    X1 --> X2
    X2 --> MLP["PointwiseMLP<br/>Conv3d(w -> 2w, 1)<br/>GELU<br/>Dropout<br/>Conv3d(2w -> w, 1)<br/>GroupNorm<br/>tanh"]
    MLP --> OUT["out = x2 + res_mlp * mlp"]
    X2 --> OUT

    MASK["branch_mask optional<br/>spec, vertical, pw, mlp"] -. scales branches when provided .-> SMIX
    MASK -. scales residual updates when provided .-> X2
    MASK -. scales residual updates when provided .-> OUT
    STATS["collect_stats optional"] -. returns layer diagnostics .-> OUT
```

## `SpectralConv2dFull`

```mermaid
flowchart TD
    X["x: [B, C, D, H, W]"] --> IG["input_gate<br/>Conv3d(C -> C, 1)<br/>GELU<br/>Conv3d(C -> C, 1)<br/>Sigmoid"]
    X --> XG["x * gate"]
    IG --> XG
    XG --> FFT["rfft2 over H,W<br/>x_ft: [B, C, D, H, Wf]"]

    DX["dx"] --> KGRID["kx = rfftfreq(W, dx)"]
    DY["dy"] --> KGRID
    KGRID --> KFEAT["frequency features<br/>kx, ky, sqrt(kx^2 + ky^2), sign(ky)"]
    KFEAT --> FMLP["freq_mlp<br/>Linear(4 -> hidden)<br/>GELU<br/>Linear(hidden -> hidden)<br/>GELU<br/>Linear(hidden -> 2)"]

    WR["learned weight_real<br/>[Cin, Cout]"] --> CW["complex spectral weights"]
    WI["learned weight_imag<br/>[Cin, Cout]"] --> CW
    FMLP --> CW

    FFT --> EINSUM["complex channel mixing<br/>einsum over Cin"]
    CW --> EINSUM
    EINSUM --> IFFT["irfft2 back to H,W"]
    IFFT --> POST["post<br/>Conv3d(Cout -> 2*Cout, 1)<br/>GELU<br/>Conv3d(2*Cout -> Cout, 1)"]
    POST --> Y["spectral branch output"]
```

## `BalancedVerticalPhysicsStack`

```mermaid
flowchart TD
    X["x: [B, C, D, H, W]"] --> PERM["permute/reshape columns<br/>[B, H*W, C, D]"]
    Z["z_scale"] --> ZCHECK["validate shape<br/>[B,1,D,H,W]"]
    ZCHECK --> ZPERM["permute/reshape columns<br/>[B, H*W, 1, D]"]

    PERM --> CHUNK["process H*W columns<br/>in chunks of 4"]
    ZPERM --> CHUNK

    CHUNK --> XI["xi: [-1, C, D]"]
    CHUNK --> ZI["zi: [-1, 1, D]"]
    ZI --> ZFEAT["z features<br/>z_scale, z_norm, dz_norm, z_span<br/>[-1, 4, D]"]

    XI --> INPROJ["in_proj<br/>Conv1d(C -> hidden, 1)<br/>GroupNorm<br/>GELU"]
    ZFEAT --> ZPROJ["z_proj<br/>Conv1d(4 -> hidden, 1)<br/>GELU"]
    INPROJ --> ADD["in_proj(xi) + z_proj(z_feat)"]
    ZPROJ --> ADD

    ADD --> MS["LiteMultiScaleVertical<br/>depthwise kernels 3,7,15<br/>pointwise mix"]
    MS --> RB1["LiteVerticalResidualBlock<br/>k=5 dilation=1"]
    RB1 --> RB2["LiteVerticalResidualBlock<br/>k=5 dilation=2"]
    RB2 --> RB3["LiteVerticalResidualBlock<br/>k=5 dilation=4"]
    RB3 --> RB4["LiteVerticalResidualBlock<br/>k=7 dilation=1"]

    RB4 --> DG["depth_gate<br/>Conv1d(hidden -> hidden, 1)<br/>GELU<br/>Conv1d(hidden -> hidden, 1)<br/>Sigmoid"]
    RB4 --> GATED["yi * depth_gate(yi)"]
    DG --> GATED
    GATED --> OUTPROJ["out_proj<br/>Conv1d(hidden -> C, 1)"]
    OUTPROJ --> COLLECT["reshape chunks back<br/>[B, H, W, C, D]"]
    COLLECT --> RESTORE["permute to<br/>[B, C, D, H, W]"]
    RESTORE --> GN["GroupNorm"]
    GN --> Y["vertical branch output"]
```

## Residual Scalars

The block learns three scalar residual strengths:

| Parameter | Initial value | Applied to |
| --- | ---: | --- |
| `res_fused` | `1.0` | gated spectral/vertical fusion |
| `res_pw` | `0.15` | pointwise convolution correction |
| `res_mlp` | `0.15` | pointwise MLP correction |

