# WH-Net: learned weak-harmonic QSM

**Goal.** From a measured field \(\phi\), WH-Net jointly estimates magnetic
susceptibility \(\chi\) and a residual background field \(h\):

\[
\phi=\mathcal A\chi+h+\epsilon, \qquad
\mathcal A\chi=\mathcal F^{-1}\!\left(D\odot\mathcal F\chi\right),
\qquad \Delta h\approx0 \text{ in the ROI}.
\]

Here, \(D\) is the dipole kernel, \(M\) is the brain mask, and \(\Lambda\) is
the voxelwise data-fidelity coefficient. The classical weak-harmonic prior
combines TV on \(\chi\) with a Laplacian penalty on \(h\). WH-Net replaces
these hand-tuned penalties with learned, branch-specific proximal mappings:

\[
\boxed{
\min_{\chi,h}\;
\frac12\left\|\Lambda^{1/2}\!\odot(\mathcal A\chi+h-\phi)\right\|_2^2
+\mathcal R_{\theta_\chi}(\chi)+\mathcal R_{\theta_h}(h)
+\iota_{\{\chi=M\odot\chi\}}(\chi)
}
\]

The CNNs are proximal **surrogates**, so this functional is a variational
interpretation; the unrolled updates below define the implemented model.

## Learned unrolled ADMM

Introduce \(y=\mathcal A\chi+h\), \(u=\chi\), and \(v=h\), with scaled duals
\(\eta_y,\eta_u,\eta_v\). Each input is normalized by its masked RMS \(s\),
giving \(\bar\phi=\phi/s\). Starting from zero, WH-Net applies \(K=55\)
shared-weight iterations:

\[
\begin{aligned}
f^k &= \mathcal A\chi^k, \\
y^{k+1} &=
\frac{\Lambda\odot\bar\phi+f^k+h^k+\eta_y^k}{\Lambda+1},\\
u^{k+1} &=P_{\theta_\chi}(\chi^k+\eta_u^k;\,k,q_\chi^k),\\
v^{k+1} &=P_{\theta_h}(h^k+\eta_v^k;\,k,r_\phi^k),\\
\chi^{k+1} &=M\odot\mathcal F^{-1}\!\left[
\frac{D\odot\mathcal F(y^{k+1}-\eta_y^k-h^k)
+\mathcal F(u^{k+1}-\eta_u^k)}{D^2+1+\varepsilon}\right],\\
f^{k+1} &=\mathcal A\chi^{k+1},\\
h^{k+1} &=\tfrac12\left(y^{k+1}-\eta_y^k-f^{k+1}
+v^{k+1}-\eta_v^k\right),\\
\eta_y^{k+1} &=\eta_y^k+f^{k+1}+h^{k+1}-y^{k+1},\\
\eta_u^{k+1} &=\eta_u^k+\chi^{k+1}-u^{k+1},\qquad
\eta_v^{k+1}=\eta_v^k+h^{k+1}-v^{k+1}.
\end{aligned}
\]

The conditioning statistics are

\[
q_\chi^k=\operatorname{RMS}_M(\chi^k-\chi^{k-1}),\qquad
r_\phi^k=\operatorname{RMS}_M(f^k+h^k-\bar\phi).
\]

Each learned mapping is a spectrally normalized 3D U-Net residual with a
bounded, iteration-dependent gain:

\[
P_{\theta_b}(x;k,q)=x+a_{\theta_b}(k,q)\,U_{\theta_b}(x),
\quad b\in\{\chi,h\},
\]

\[
a_{\theta_b}=0.25\frac{\operatorname{softplus}(g_{\theta_b}(k/60,\log(1+q)))}
{1+\operatorname{softplus}(g_{\theta_b}(k/60,\log(1+q)))}.
\]

Finally, \((\hat\chi,\hat h)=s(\chi^K,h^K)\). The masked susceptibility step
is a projected quadratic update, rather than an exact support-constrained
quadratic solve.

## Training objective

Four distributed iterates receive deep supervision, with 70% of the weight on
the final selected iterate:

\[
\mathcal L=\sum_{k\in\mathcal K}\omega_k\left[
\lambda_\chi\,\mathcal L_{\chi}^{k}
+\lambda_h\,\mathcal L_h^{k}
+\lambda_\nabla\,\mathcal L_\nabla^{k}
+\lambda_{\mathrm{WH}}\,\mathcal L_{\mathrm{WH}}^{k}\right],
\]

where \(\mathcal L_\chi\) and \(\mathcal L_h\) are masked, energy-normalized
MSEs, \(\mathcal L_\nabla\) is the normalized \(L_1\) susceptibility-gradient
error, and

\[
\mathcal L_{\mathrm{WH}}
=\operatorname{SmoothL1}\!\left(
\Delta\frac{\hat h}{\operatorname{RMS}_M(h^\star)},0\right)
\]

is evaluated only where the complete \(3\!\times\!3\!\times\!3\) Laplacian
stencil lies inside the mask. The implemented weights are
\((\lambda_\chi,\lambda_h,\lambda_\nabla,\lambda_{\mathrm{WH}})
=(1,0.25,0.1,0.1)\); the harmonic term ramps in over the first five epochs.

## Dataset and training configuration

- **Inputs:** 651 preprocessed \(160^3\) susceptibility volumes from 161 source
  subjects (105 subjects with three copies and 56 subjects with six copies).
- **Subject-safe split:** the split manifest contains 585 volumes/145 subjects
  for training and 66 volumes/16 subjects for validation; all copies of a
  subject remain in one partition. Evaluation uses one stored volume and three
  fixed synthetic realizations per validation subject (48 cases).
- **On-device synthesis:** random axis permutations/flips, susceptibility scale
  \(0.5{-}3\), dipole forward simulation, exterior susceptibility sources
  outside a dilated mask, spatially varying reliability, and RMS-defined
  Gaussian noise with SNR sampled from \(40{-}100\).
- **Optimization:** 50 epochs and 5,849 optimizer updates; unrolling depth is
  ramped from 12 to 30 iterations, followed by 10 epochs at 55 iterations.
  Both proximal branches are trained jointly with EMA checkpoint selection by
  subject-mean susceptibility NRMSE at 55 iterations.

**Inference:** one pretrained checkpoint and a fixed 55-step schedule require
no case-specific selection of TV, weak-harmonic, or ADMM penalty parameters.
