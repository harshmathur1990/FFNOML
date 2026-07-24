# 3D NLTE Radiative Transfer with Factorized Fourier Neural Operators

## What This Paper Should Contain

### Suggested Paper Structure

1. **Abstract**
   - State the bottleneck: full 3D NLTE radiative transfer is accurate but extremely expensive.
   - State the proposal: a factorized Fourier neural operator that maps 3D atmospheric states to NLTE departure coefficients.
   - State why departure coefficients are the right target: they are reusable for spectral synthesis and not tied to one viewing angle or wavelength.
   - Summarize the main validation: departure-coefficient errors, height-dependent error envelopes, spectra/intensity diagnostics if available, speedup, and generalization tests.

2. **Introduction**
   - Motivate 3D NLTE synthesis for chromospheric diagnostics.
   - Explain why iterative solvers such as Multi3D are costly.
   - Position prior ML work:
     - SunnyNet: local 3D CNN windows for hydrogen NLTE populations.
     - Graph-network approach: flexible 1D graph representation for departure coefficients and inversions.
   - Identify the gap: fixed local CNN windows limit nonlocal context, while graph methods are naturally flexible but were demonstrated mainly for 1D columns/inversions. Your method learns a field-to-field operator over 3D simulation volumes/patches.
   - End with the contributions.

3. **NLTE Departure Coefficients and Learning Target**
   - Define LTE and NLTE populations and departure coefficients, \(b_i=n_i^\mathrm{NLTE}/n_i^\mathrm{LTE}\).
   - Explain that the network predicts \(\log_{10} b_i\) for all levels.
   - Explain why this target is useful: once \(b_i\) is known, source functions and opacities can be reconstructed for downstream formal synthesis.

4. **Data**
   - Describe Bifrost/MULTI3D snapshots and the active atom/model atom.
   - List input channels: \(\log_{10}T, v_x, v_y, v_z, \log_{10}n_e, \log_{10}\rho\).
   - List output channels: six hydrogen departure-coefficient levels in the current configuration.
   - Explain train/validation/test splits by snapshot and simulation family.
   - Describe patching, multiscale XY extraction, stride, normalization, and native \(z\)-scale handling.

5. **Factorized Fourier Neural Operator**
   - Explain the operator-learning view: learn a mapping between fields rather than a pointwise or columnwise regression.
   - Describe the factorization:
     - horizontal spectral branch: full 2D FFT mixing in \(x,y\) at each depth, aware of \(dx,dy\);
     - vertical branch: coordinate-conditioned 1D stack along the nonuniform \(z\) grid;
     - adaptive gated fusion of horizontal and vertical updates;
     - pointwise residual corrections.
   - Include the model-flow figure from `docs/ffno_full_forward.png` and block figures.
   - Discuss why the factorization is physically motivated: horizontal radiation coupling is spatially nonlocal, while atmospheric stratification and source-function structure are strongly vertical and nonuniform.

6. **Training Objective and Optimization**
   - Describe the data loss on normalized \(\log_{10} b_i\).
   - Describe gradient/regularization terms if used.
   - Describe the physics-inspired source-function regularizer if it is part of the final run.
   - Give optimizer, learning rate, scheduler, batch size, model width/layers, dropout, training epochs, GPU setup, and early stopping.

7. **Experiments**
   - Validation on held-out snapshots from the same simulation.
   - Prediction on additional snapshots not used during training.
   - If available, cross-simulation tests.
   - Branch ablation: spectral, vertical, pointwise, MLP components.
   - Runtime comparison: FFNO inference versus MULTI3D.
   - Downstream synthesis: compare H-alpha or relevant line profiles/intensity maps if available.

8. **Results**
   - Departure-coefficient error distributions.
   - Height-dependent relative-error and log-ratio envelopes.
   - Spectral/intensity agreement.
   - Runtime/speedup.
   - Ablation results showing the contribution of spectral and vertical branches.

9. **Discussion**
   - Strengths: field-level operator, global horizontal context within patches/cubes, native \(z\)-scale awareness, reusable departure coefficients, fast inference.
   - Limitations: dependence on simulation coverage, possible degradation for out-of-distribution magnetic/thermal states, approximate nature of predictions, current atom/snapshot scope.
   - Comparison with SunnyNet and graph networks.
   - Future work: Ca II and multi-atom training, PRD-sensitive diagnostics, larger cross-simulation training sets, uncertainty estimates, coupling to inversion codes.

10. **Conclusions**
   - Restate the method and the core result.
   - Make a crisp claim: FFNOs are a promising surrogate for expensive 3D NLTE population calculations and can enable fast synthesis on large 3D atmospheres.

## Figure And Table Plan

- **Figure 1:** Overview of the surrogate workflow: Bifrost atmosphere -> MULTI3D training labels -> FFNO prediction -> formal synthesis.
- **Figure 2:** FFNO architecture diagram, using `docs/ffno_full_forward.png`.
- **Figure 3:** Balanced FFNO block, using `docs/ffno_balanced_block.png`.
- **Figure 4:** Height-dependent relative-error envelopes for a held-out snapshot.
- **Figure 5:** Log-ratio error envelopes for the same snapshot.
- **Figure 6:** Example maps or profiles from downstream spectral synthesis.
- **Figure 7:** Branch-ablation bar plot from validation diagnostics.
- **Table 1:** Simulation/snapshot split.
- **Table 2:** Model and training hyperparameters.
- **Table 3:** Summary metrics: median error, 68% interval, 95% interval, validation loss, runtime.

## First Draft

### Abstract

Three-dimensional non-local thermodynamic equilibrium (3D NLTE) radiative-transfer calculations are essential for realistic synthesis of chromospheric diagnostics, but their iterative solution is computationally expensive enough to limit their routine use on large numerical simulations. We present a neural surrogate for 3D NLTE population calculations based on a factorized Fourier neural operator (FFNO). The model maps the thermodynamic and velocity structure of a 3D atmosphere to the logarithmic NLTE departure coefficients of a hydrogen model atom. Instead of predicting emergent spectra directly, the network predicts the atomic state variables needed by a subsequent formal solution, making the surrogate independent of viewing angle and reusable for multiple spectral transitions within the model atom.

The proposed architecture factorizes the 3D operator into a metric-aware horizontal spectral branch and a coordinate-conditioned vertical branch. The horizontal branch performs Fourier mixing over each geometrical height plane using the physical grid spacing, while the vertical branch processes each atmospheric column on its native, nonuniform \(z\) scale. The two branches are combined through adaptive gating and residual pointwise corrections. Trained on MULTI3D/Bifrost calculations, the model predicts six hydrogen departure-coefficient levels from \(\log_{10}T\), velocity components, \(\log_{10}n_e\), and \(\log_{10}\rho\). On held-out snapshots, the median departure-coefficient error remains close to zero across most heights, with the largest spreads concentrated in the chromosphere and transition-region heights where departures from LTE and spatial intermittency are strongest. The resulting inference cost is orders of magnitude smaller than a full 3D NLTE solve. These results suggest that factorized neural operators can provide fast, physically useful approximations to 3D NLTE population calculations for large-scale spectral synthesis workflows.

### 1. Introduction

Forward modelling of stellar and solar spectra from 3D radiation-magnetohydrodynamic simulations has become a central tool for interpreting observations of structured atmospheres. In the solar chromosphere, many of the most informative diagnostics form far from local thermodynamic equilibrium. The level populations and radiation field must then be obtained from the coupled radiative-transfer and statistical-equilibrium equations. This coupling is nonlinear and nonlocal: radiation emitted in one region can influence the excitation and ionization state elsewhere, particularly where photon mean free paths become large. Full 3D NLTE calculations therefore remain expensive, even on modern high-performance computing systems.

The cost of 3D NLTE synthesis is especially restrictive when one wants to process many snapshots, explore parameter variations, or use synthetic spectra in inversion and data-assimilation workflows. A natural alternative is to learn a fast approximation from accurate NLTE calculations. Previous work has shown that neural networks can reproduce useful parts of the NLTE solution. SunnyNet used convolutional neural networks to predict hydrogen NLTE populations from local 3D windows and demonstrated large speedups for H-alpha synthesis. Graph-network approaches have predicted departure coefficients in 1D atmospheric columns and shown that such surrogates can accelerate both synthesis and inversions. These studies establish two important lessons: departure coefficients are a flexible learning target, and approximate population predictions can still produce accurate spectra because radiative-transfer observables depend on combinations and ratios of populations.

This work explores a different surrogate architecture: a factorized Fourier neural operator for 3D NLTE radiative transfer. Rather than learning a fixed-window local mapping or a purely columnwise mapping, we learn a field-to-field operator from atmospheric variables to departure coefficients. The architecture is designed around two features of the physical problem. First, horizontal radiative coupling is nonlocal and benefits from spectral mixing over extended spatial scales. Second, the vertical dimension has special structure: stratification, optical-depth variation, shocks, and transition-region gradients make the native height coordinate physically important and generally nonuniform. Our FFNO treats these directions differently, combining a horizontal Fourier branch with a coordinate-aware vertical branch.

The main contributions of this paper are:

1. We formulate 3D NLTE population prediction as an operator-learning problem for departure coefficients on 3D atmospheric fields.
2. We introduce a factorized architecture with metric-aware horizontal spectral convolutions and native-\(z\) vertical processing.
3. We train and validate the model on MULTI3D/Bifrost hydrogen calculations.
4. We evaluate the method using departure-coefficient error envelopes, log-ratio errors, runtime, and component ablations.

### 2. NLTE Departure Coefficients

For a model atom with energy levels indexed by \(i\), the NLTE departure coefficient is

\[
b_i = \frac{n_i^\mathrm{NLTE}}{n_i^\mathrm{LTE}},
\]

where \(n_i^\mathrm{NLTE}\) is the population obtained from the NLTE statistical-equilibrium solution and \(n_i^\mathrm{LTE}\) is the corresponding LTE population. Values \(b_i=1\) indicate LTE populations, while deviations from unity encode the effect of the nonlocal radiation field and non-equilibrium excitation/ionization balance.

We train the network to predict

\[
y_i = \log_{10} b_i,
\]

for every spatial cell and atomic level. The logarithmic target is numerically convenient because departure coefficients can span several orders of magnitude. It also makes relative errors in the population ratio more symmetric during optimization. Once the departure coefficients are predicted, they can be converted back to linear space and used with LTE populations to reconstruct approximate NLTE populations for formal spectral synthesis.

This choice deliberately separates the expensive NLTE population problem from the comparatively cheaper formal solution. A network trained to predict an emergent intensity profile would be tied to a particular spectral line, wavelength sampling, and viewing direction. In contrast, predicted departure coefficients are atomic state variables. They can be reused for any transition represented by the model atom, subject to the assumptions of the downstream synthesis.

#### 2.3. From Departure Coefficients to Spectral Synthesis

After inference, the network output is transformed from logarithmic to linear departure coefficients and combined with the LTE populations of the same atmosphere,

\[
n_i^\mathrm{pred} = b_i^\mathrm{pred} n_i^\mathrm{LTE}.
\]

These predicted populations provide the atomic state used by a subsequent formal solution of the radiative-transfer equation. In this workflow, the neural network does not replace the final synthesis step. Instead, it replaces the costly iterative part of the NLTE calculation in which the radiation field and statistical-equilibrium equations are solved until the level populations converge. The formal solver can then use the reconstructed populations to compute opacities, emissivities, source functions, and emergent intensities for the desired transition, wavelength grid, and viewing direction.

This separation is useful because many observables depend on ratios or combinations of level populations rather than on a single population value alone. For example, the line opacity is mainly controlled by the lower-level population, while the line source function depends on the relation between the upper and lower levels. Predicting the set of departure coefficients therefore gives the synthesis code access to the quantities that determine both absorption and emission. Errors in individual levels can still matter, but evaluating the surrogate at the population level keeps the method physically interpretable and allows the final validation to be performed directly on synthetic spectra.

The approach also makes the learned surrogate more reusable than a model trained to output one intensity profile directly. A spectrum-prediction model would be tied to the training choice of line, angle, wavelength sampling, and instrumental setup. Departure coefficients are independent of these choices once the model atom and atmosphere are fixed. The same predicted coefficients can therefore be used for different rays, multiple transitions within the hydrogen atom, and different post-processing choices in the formal synthesis. This is the main reason we use departure coefficients as the learning target rather than emergent intensity.

### 3. Data

The training labels are generated from MULTI3D calculations on Bifrost atmospheres. In the present configuration, the active atom is hydrogen with six output levels. The input tensor contains six atmospheric channels:

\[
\left[\log_{10}T,\ v_x,\ v_y,\ v_z,\ \log_{10}n_e,\ \log_{10}\rho\right].
\]

The target tensor contains

\[
\left[\log_{10}b_1,\ldots,\log_{10}b_6\right].
\]

The data pipeline builds HDF5 datasets from atmosphere files, mesh files, and MULTI3D output directories. Input and target channels are normalized using statistics computed from the training set. Training examples are extracted as multiscale horizontal patches with a configurable patch size and stride. The current configuration uses \(40\times40\) horizontal patches, stride 20, and multiple spatial scales. Each patch preserves the native vertical grid of its source snapshot and carries its corresponding \(z\)-scale, \(dx\), and \(dy\). This is important because the model uses physical grid spacing in the spectral branch and absolute/relative vertical coordinates in the vertical branch.

The current split uses selected snapshots from Bifrost simulation families for training and validation. The final paper should include a table listing the simulation name, snapshot number, role, grid size, physical extent, and atom labels available from MULTI3D. A second table should list any prediction-only snapshots used for qualitative and quantitative tests.

### 4. Factorized Fourier Neural Operator

The network learns the mapping

\[
\mathcal{G}_\theta:
\left(X(\mathbf{r}), z(\mathbf{r}), dx, dy\right)
\mapsto
Y(\mathbf{r}),
\]

where \(X\) is the 3D atmospheric state, \(Y\) is the 3D field of logarithmic departure coefficients, and \(\mathbf{r}=(x,y,z)\). The model first lifts the input channels into a latent width using a \(1\times1\times1\) convolution, group normalization, and a GELU activation. The latent field is then processed by a stack of factorized FFNO blocks before projection back to the six output channels.

Each block contains two main branches.

The **horizontal spectral branch** applies a two-dimensional real FFT over the horizontal dimensions at each depth. Unlike a standard Fourier layer with fixed learned modes, this implementation uses the full resolved horizontal spectrum and conditions the complex spectral weights on physical frequency features derived from \(dx\) and \(dy\). The frequency features include \(k_x\), \(k_y\), radial wavenumber, and the sign of \(k_y\). A small MLP maps these features to complex modulation factors, allowing the same architecture to respond to different physical grid spacings. The spectral branch is therefore designed to represent broad horizontal coupling efficiently.

The **vertical branch** processes each \((x,y)\) column independently but conditions the computation on the actual \(z\) coordinate. For each column, the branch constructs vertical coordinate features including absolute \(z\), normalized local depth, normalized local spacing, and total vertical span. These features are projected into the latent space and added to the projected column state. The column is then processed by multiscale and dilated depthwise-separable convolutions, followed by a learned depth gate. This branch gives the model a structured representation of stratification, transition-region gradients, and nonuniform vertical sampling.

The outputs of the spectral and vertical branches are normalized, activated, and fused with an adaptive gate. The gate sees the incoming latent state and both branch outputs, then produces separate weights for the spectral and vertical contributions. A learned fusion network combines the gated branches, and residual pointwise convolution/MLP corrections refine the latent field. The block therefore lets the model balance horizontal nonlocal mixing against vertical atmospheric structure at each layer.

This factorization is motivated by the physical anisotropy of solar-atmosphere radiative transfer. Horizontal transport and fibril-scale coupling require extended spatial context, while vertical source-function behavior is tied to steep stratification and the depth scale. A monolithic 3D convolution treats these directions uniformly and has a finite receptive field. A purely columnwise method misses horizontal coupling. The FFNO aims for a middle ground: nonlocal horizontal mixing plus explicit vertical coordinate awareness.

### 5. Training Objective

The baseline loss is a mean-squared error between predicted and target normalized \(\log_{10} b_i\). Because the network predicts logarithmic departure coefficients, the loss emphasizes multiplicative agreement in the linear departure coefficients rather than absolute population differences.

Additional loss terms can be included to regularize physically relevant structure. The code supports a gradient loss on the full predicted field and a source-function smoothness term based on the departure-coefficient ratio entering the line source function. For a transition between lower level \(l\) and upper level \(u\), the source-function dependence can be expressed through the combination

\[
\log \left(\frac{b_l}{b_u}\right) + \frac{\Delta \chi}{k_B T}.
\]

This quantity controls the denominator of the line source function in complete redistribution. A multiscale vertical curvature penalty can discourage unphysical zig-zag structure in the derived source function. In the final manuscript, this section should state exactly which loss terms were active in the reported checkpoint and give their weights.

The present model configuration uses a width of 48, four FFNO blocks, AdamW optimization, cosine learning-rate annealing, gradient clipping, and checkpointed blocks to reduce memory use. Training is performed on grouped native-depth patch datasets with batch size one. The final paper should report the number of epochs, training time, hardware, final validation loss, and any early-stopping or checkpoint-selection criterion.

### 6. Validation Strategy

The surrogate should be evaluated at three levels.

First, we compare predicted and MULTI3D departure coefficients directly. Useful diagnostics include per-level distributions of \(\log_{10}(b_\mathrm{NN}/b)\), relative-error envelopes as a function of height, and summary intervals such as the median, 68% range, and 95% range. These diagnostics reveal whether the model is biased and where in the atmosphere the prediction is most difficult.

Second, we validate downstream spectral consequences. Departure-coefficient errors do not translate linearly into emergent-intensity errors. The source function depends strongly on ratios of level populations, while the opacity is especially sensitive to the lower-level population. Therefore, formal synthesis from the predicted populations is the decisive test. The paper should compare line profiles, spatially averaged spectra, and intensity maps against synthesis using MULTI3D populations.

Third, we test model components. Branch ablations can quantify how much the horizontal spectral branch, vertical branch, pointwise branch, and MLP branch contribute to validation accuracy. This is important because the main methodological claim is not just that a neural network works, but that a factorized operator architecture is appropriate for 3D NLTE transfer.

### 7. Results

On the held-out `en024048_hion_385` snapshot, the current error-envelope figures show that the median relative error is close to zero for all six hydrogen levels across most of the plotted height range. The 68% and 95% intervals widen most strongly around the lower-to-mid chromosphere, where departures from LTE and sharp thermal/ionization structure make the target more intermittent. In the plotted log-ratio diagnostics, the median \(\log_{10}(b_\mathrm{NN}/b)\) remains near zero, indicating little systematic multiplicative bias. The widest log-ratio envelopes occur near the same chromospheric heights, especially for excited levels.

These trends are encouraging for spectral synthesis. A near-zero median log-ratio error means that the model is not simply shifting departure coefficients upward or downward. The concentration of larger errors in localized height ranges is expected for a data-driven approximation to an intermittent NLTE problem. The final manuscript should quantify these statements with exact median and percentile values extracted from the prediction files, not just the plotted envelopes.

The model should also be compared across multiple snapshots. The available prediction plots for `en024048_hion_386`, `en024048_hion_465`, and `ch012006_795` can support a section on generalization. Same-family snapshots should be reported separately from different-family simulations, because prior work shows that ML NLTE surrogates usually perform best when training and testing atmospheres come from similar simulation distributions.

Runtime should be reported as wall-clock time for three stages: dataset preparation, training, and inference. The physically relevant comparison is inference time for a trained network versus a full MULTI3D population calculation for the same atmosphere and atom. The paper should include GPU type, CPU type, grid size, patch/tiled inference settings, and I/O time. If possible, report both end-to-end inference time including HDF5 I/O and pure neural-network forward time.

### 8. Discussion

The FFNO surrogate differs from earlier CNN-window approaches in that the network is designed as a field operator. The spectral branch allows information to mix across the full horizontal extent of each patch or tile, while the vertical branch preserves the special role of atmospheric stratification. This should be advantageous for chromospheric lines whose morphology reflects genuinely 3D radiative coupling. At the same time, the architecture is still much cheaper than solving the coupled NLTE equations iteratively.

Compared with graph-network approaches, the present method is less general with respect to arbitrary mesh connectivity but more directly optimized for structured 3D simulation data. Bifrost/MULTI3D atmospheres are naturally represented as rectilinear grids with known physical spacing. Fourier layers exploit this structure efficiently. The vertical branch adds some of the coordinate awareness that motivates graph methods, but without the cost of many message-passing iterations over all grid points.

The main limitation is distribution dependence. The model can only learn NLTE behavior represented in its training set. Rare flaring states, unusual magnetic configurations, different spatial resolutions, or atoms with different radiative coupling may require additional training data or fine-tuning. The model is also approximate by construction. It should be used as a fast surrogate for synthesis, survey calculations, initialization of full NLTE solvers, or exploratory modelling, rather than as a replacement for benchmark NLTE calculations in regimes where high precision is essential.

Future work should extend the method to additional atoms such as Ca II, test multi-atom outputs, and evaluate the predicted departure coefficients through full spectral synthesis. Another natural direction is uncertainty estimation, either through ensembles, dropout, or calibration against validation residuals. Finally, because the network is differentiable, it could be coupled to inversion workflows in which departure coefficients are updated rapidly as atmospheric parameters change.

### 9. Conclusions

We presented a factorized Fourier neural operator for approximating 3D NLTE departure coefficients from 3D atmospheric simulations. The method predicts logarithmic hydrogen departure coefficients from temperature, velocity, electron-density, and mass-density fields. Its architecture combines metric-aware horizontal Fourier mixing with coordinate-conditioned vertical processing, reflecting the anisotropic structure of the radiative-transfer problem.

Initial validation against MULTI3D labels shows small median bias in the predicted departure coefficients and physically interpretable error growth in chromospheric height ranges. Because the model predicts departure coefficients rather than spectra directly, its outputs can be used in downstream formal synthesis for different transitions and viewing directions supported by the model atom. The inference cost is expected to be orders of magnitude lower than a full 3D NLTE population solve, making the approach promising for rapid synthesis over large simulation volumes and time series.

The final version of this paper should strengthen the quantitative claims with exact percentile metrics, spectra/intensity comparisons, runtime measurements, and ablation results. With those additions, the manuscript can make a clear case that factorized neural operators are a practical and physically motivated route toward fast 3D NLTE radiative-transfer surrogates.

## Immediate To-Do List Before Turning This Into A Manuscript

1. Extract exact metrics from prediction HDF5 files:
   - median relative error by level;
   - 68% and 95% intervals by level;
   - median and percentile ranges of \(\log_{10}(b_\mathrm{NN}/b)\).
2. Add a table of simulations and snapshots:
   - train, validation, and prediction-only.
3. Run or summarize branch ablations from `pipeline.py --test`.
4. Measure inference runtime on one full atmosphere:
   - with I/O;
   - neural forward only;
   - tiled versus full-cube if both are available.
5. Add downstream synthesis:
   - H-alpha profile comparisons;
   - line-core and wing intensity maps;
   - spatially averaged spectra.
6. Decide the target journal style:
   - A&A style if following SunnyNet closely;
   - ApJ style if following the graph-network paper more closely.
