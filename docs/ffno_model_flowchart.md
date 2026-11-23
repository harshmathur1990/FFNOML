# FFNO3D Model Flowchart

Source: [`models/ffno_model.py`](../models/ffno_model.py)

## Full Forward Pass

```mermaid
flowchart TD
    X["x: [B, in_channels, D, H, W]"] --> LIFT["lift<br/>Conv3d(in_channels -> width, 1)<br/>DepthWeightedGroupNorm<br/>GELU"]
    Z["z_scale: [B, D, H, W] or [B, 1, D, H, W]"] --> BLOCKS
    DX["dx"] --> BLOCKS
    DY["dy"] --> BLOCKS
    LIFT --> H0["hidden: [B, width, D, H, W]"]
    H0 --> BLOCKS["repeat n_layers x<br/>FFNOBlock3dBalanced"]
    BLOCKS --> HEAD1["proj1<br/>Conv3d(width -> 2*width, 1)<br/>GELU"]
    HEAD1 --> HEAD2["proj2<br/>Conv3d(2*width -> head_channels, 1)"]
    HEAD2 --> HO["head_output(head, x)<br/>level-fraction head: see below"]
    X -. T, ne, rho .-> HO
    HO --> Y["output: normalised log10 n, [B, out_channels, D, H, W]"]

    TRAIN{"training and<br/>checkpoint_blocks and<br/>not collect_stats?"} -. wraps lift/block/head .-> LIFT
    TRAIN -. wraps lift/block/head .-> BLOCKS
    TRAIN -. wraps lift/block/head .-> HEAD1
```

`head_channels` is `out_channels + n_species`: one departure value per level and one total correction per species.

This is the final model (zaug, `config.py`'s defaults, width 96), the only one this branch builds. Every GroupNorm is a
`DepthWeightedGroupNorm` (statistics weighted by each level's share of the column height), each block's vertical
branch is `HeightSpectralConv` (a spectral convolution along height, low-rank mixing of rank 8, column cutoff), and
the fusion is pointwise.

## Level-fraction head with departure coefficients

```mermaid
flowchart TD
    HEAD["proj2 output, per species:<br/>delta_1..delta_L (departures), u (total)"]
    X["inputs x: log10 T, log10 ne, log10 rho<br/>(un-normalised with mean_X, std_X)"]
    X --> LTE["multi3d_lte_log_weights (float64)<br/>ln f*_i: MULTI3D calc_lte_populations<br/>Saha-Boltzmann, Debye lowering, MULTI3D atom data"]
    HEAD --> ADD["z_i = ln f*_i + delta_i<br/>delta_i = ln b_i + constant"]
    LTE --> ADD
    ADD --> SM["log_softmax over the species' levels<br/>share_i = f*_i b_i / sum_j f*_j b_j"]
    HEAD --> S["s = 6 tanh(u / 6)"]
    X --> TOT["log10 n_total = log10(rho / m) + s"]
    S --> TOT
    SM --> N["log10 n_i = log10 n_total + log10 share_i"]
    TOT --> N
    N --> NORM["(log10 n - mean_Y) / std_Y"]
```

δ = 0 gives MULTI3D's LTE populations; the network learns the departure coefficients (up to a constant per species)
and the total's small offset from ρ / m. The head has no parameters of its own.

## `FFNOBlock3dBalanced`

```mermaid
flowchart TD
    X["x: [B, width, D, H, W]"] --> RES["residual = x"]

    X --> SPEC["SpectralConv2dFull(x, dx, dy)"]
    DX["dx"] --> SPEC
    DY["dy"] --> SPEC
    SPEC --> NS["DepthWeightedGroupNorm<br/>GELU<br/>optional spec_dropout"]

    X --> VERT["HeightSpectralConv(x, z_scale)"]
    Z["z_scale"] --> VERT
    VERT --> NV["DepthWeightedGroupNorm<br/>GELU<br/>optional vertical_dropout"]

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
    CAT --> FUSE["fuse<br/>Conv3d(2w -> 2w, 1)<br/>GELU<br/>Conv3d(2w -> w, 1)"]
    SMIX --> AVG["0.5 * (spec_mix + vert_mix)"]
    VMIX --> AVG
    FUSE --> FADD["fused + average branch mix"]
    AVG --> FADD
    FADD --> NF["DepthWeightedGroupNorm<br/>GELU"]

    RES --> X1["x1 = residual + res_fused * fused"]
    NF --> X1
    X1 --> PW["pw<br/>Conv3d(w -> 2w, 1)<br/>GELU<br/>Conv3d(2w -> w, 1)<br/>DepthWeightedGroupNorm"]
    PW --> X2["x2 = x1 + res_pw * pw"]
    X1 --> X2
    X2 --> MLP["PointwiseMLP<br/>Conv3d(w -> 2w, 1)<br/>GELU<br/>Dropout<br/>Conv3d(2w -> w, 1)<br/>DepthWeightedGroupNorm"]
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

With `freq_mix_rank` 8 a second MLP of the same frequency features gives each wavenumber its own low-rank channel
mixing U(k) V(k), added to the complex channel mixing (U starts at zero).

## `HeightSpectralConv`

```mermaid
flowchart TD
    X["x: [B, C, D, H, W]"] --> G["x * input_gate(x)<br/>Conv3d 1x1, GELU, Conv3d 1x1, sigmoid"]
    Z["z_scale: each column's own heights [Mm]"] --> FW["height_fourier_weights<br/>exact transform of the piecewise-linear column"]
    Z --> ZP["z_proj(z): pointwise height term"]
    G --> ADD["x * gate + z_proj(z)"]
    ZP --> ADD
    ADD --> FT["X_m = integral x(z) exp(-i k_m z) dz<br/>k_m = 2 pi m / L, m = 0..L/min_wavelength<br/>(L = 40 Mm, 1001 modes)"]
    FW --> FT
    FT --> W["Y_m = W(k_m) X_m<br/>W = complex C x C matrix x freq_mlp(t(k))<br/>+ low-rank U(k) V(k), rank 8<br/>t(k) on the fixed scale z_feature_wavelength = 0.3 Mm"]
    Z --> CUT["column_cutoff: taper from 2 pi / (3 dz) to 0 at pi / dz<br/>dz = the column's median spacing"]
    CUT --> W
    W --> INV["y(z_j) = (Y_0 + 2 Re sum Y_m exp(i k_m z_j)) / L<br/>at the column's own heights"]
    INV --> POST["post: Conv3d C -> 2C -> C, 1x1"]
    POST --> Y["vertical branch output"]
```

## Residual Scalars

The block learns three scalar residual strengths:

| Parameter | Initial value | Applied to |
| --- | ---: | --- |
| `res_fused` | `1.0` | gated spectral/vertical fusion |
| `res_pw` | `0.5` | pointwise convolution correction |
| `res_mlp` | `0.5` | pointwise MLP correction |

