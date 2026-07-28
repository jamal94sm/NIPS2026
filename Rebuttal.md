## Response to AC

We thank the AC for the constructive meta-review and for highlighting the three priority areas. Below we summarize how each was addressed across our responses to the three reviewers.

### 1. Exact Details of the Released Dataset

- **Reviewer #2 (Point 1 – Dataset Limitations):** We provided a direct scale comparison table against CASIA-MS, XJTU-UP, and MPD-v2 (hands, images, devices, age range, paired scanner–smartphone availability), and explained the IRB-approved constraints (fixed scanner location, private indoor collection sites) that shaped recruitment and demographic composition. We reported concrete diversity figures (40.78% female, 11 self-reported ethnic groups) and explicitly limited our fairness/generalization claims accordingly.
- **Reviewer #2 (Point 2 – Ethical/Privacy Documentation):** We detailed the access-control mechanism: encrypted institutional storage, mandatory EULA prohibiting re-identification or commercial use, and time-limited secure download links.
- **Reviewer #2 (Point 5 – What Users Receive):** We itemized the full release package — raw images, extracted RoIs, keypoint coordinates, anonymized metadata (age group, ethnicity group, gender, device model), official data splits, and supporting files (`LICENSE_EULA.md`, `CITATION.cff`, Croissant metadata, manifests) — contrasting this with prior datasets that release only raw images or only RoIs.
- **Reviewer #3 (Point 1):** Reused and cross-referenced the fuller dataset-detail answer given to Reviewer #2 due to space constraints, avoiding duplication while ensuring the same information is accessible.

### 2. Statistical Robustness of Results

- **Reviewer #1 / #3 (Imbalance Effects):** We introduced two controlled experiments with identity-level bootstrap 95% CIs ($B=1000$): a training-time imbalance test (Mode A vs. Mode B) and an inference-time gallery-composition test (Mode 1 vs. Mode 2), showing that Rank-1 accuracy is far more sensitive to acquisition imbalance than EER, with non-overlapping CIs supporting the effect's significance.
- **Reviewer #1 / #3 (Domain Shift Quantification):** We reported five complementary distributional-shift metrics (MMD, PAD, FFD, KID, SWD) as mean ± std across all sub-domain pairs per dataset, cross-checked with two different embedding backbones (DINOv2, ResNet-50) to confirm the ranking is not an artifact of a single feature space or metric's small-sample bias.
- **Reviewer #2 (Point 3 – Test Set Size Effects):** This was the most direct treatment of statistical robustness: we compared 3-fold vs. 5-fold cross-validation and ratio=0.5 vs. ratio=0.8 bootstrap CIs across all 12 test domains, showing consistent widening of variance/CI as the held-out identity set shrinks. We committed to releasing the exact fold/split files as an official benchmark protocol so future comparisons use identical identity assignments rather than ad hoc splits.

### 3. Annotation Quality

- **Reviewer #2 (Point 4) / Reviewer #3 (Point 3 – RoI Extraction Pipeline):** We described the custom annotation tool with live-preview quality control, the keypoint-based RoI definition process, and the exclusion criteria for low-quality images (blur, occlusion). We reported a repeatability check (10 images re-annotated over 10 iterations by the sole annotator) showing high spatial consistency, and noted that RoI extraction was performed manually specifically to eliminate automated-extraction error as a confound in benchmarking.






### reviewer #1


We thank the reviewer for taking the time to write such a detailed review and for analyzing the paper in-depth. 
Below, we address each concern point-by-point and describe the manuscript changes we will make in the camera-ready version.

>**1: Including more metrics**

We appreciate this suggestion. We actually have included computational complexity metrics in the manuscript. Table 2 details both the Parameter count and the inference cost (in GFLOPs) for an input resolution of $112 \times 112$ across all 12 evaluated baseline models. 

In the table below, we extend the Table 2 (Computational Complexity) by including the required time for training (per 32-sample batch) and inference (per 100 samples) for all baselines using NVIDIA RTX A6000(48GB). As an initial hardware-normalized analysis, we consider a low-end-class compute scenario with an assumed sustained throughput of 50 GFLOP/s and a flagship-class scenario with 500 GFLOP/s. We estimate the compute-only inference latency and throughput from the reported GFLOPs using

$$
t_{\mathrm{est}} = \frac{\mathrm{GFLOPs/image}}{\mathrm{effective\ throughput\ (GFLOP/s)}} \times 1000 \quad \mathrm{ms/image},
$$

$$
\mathrm{Throughput}_{\mathrm{est}} = \frac{\mathrm{effective\ throughput\ (GFLOP/s)}}{\mathrm{GFLOPs/image}} \quad \mathrm{images/s}.
$$


**Table 2: Computational complexity comparison of baselines using various metrics**
| Model | Params | GFLOPs | Low-end Latency (ms/img) | Low-end Speed (img/s) | Flagship Latency (ms/img) | Flagship Speed (img/s) | Train (ms/batch) | Inference (ms/100) |
|---|---|---|---|---|---|---|---|---|
| CompNet | 3.27M | 0.735 | 14.70 | 68.0 | 1.47 | 680.3 | 324.45 | 331.91 |
| PPNet | 3.53M | 0.735 | 14.70 | 68.0 | 1.47 | 680.3 | 321.61 | 358.69 |
| CCNet | 20.57M | 2.131 | 42.62 | 23.5 | 4.26 | 234.6 | 781.44 | 1722.66 |
| CO3Net | 20.57M | 2.131 | 42.62 | 23.5 | 4.26 | 234.6 | 831.50 | 1563.01 |
| SF2Net | 13.08M | 2.655 | 53.10 | 18.8 | 5.31 | 188.3 | 796.28 | 687.42 |
| PalmBridge | 3.53M | 0.735 | 14.70 | 68.0 | 1.47 | 680.3 | 318.31 | 412.11 |
| TSCAN | 11.31M | 0.486 | 9.72 | 102.9 | 0.97 | 1028.8 | 89.75 | 604.96 |
| GIFT | 11.24M | 0.486 | 9.72 | 102.9 | 0.97 | 1028.8 | 76.80 | 647.60 |
| ConvNeXtV2-T | 27.87M | 1.067 | 21.34 | 46.9 | 2.13 | 468.6 | 607.95 | 1818.77 |
| DINOv2-S/14 | 22.06M | 1.398 | 27.96 | 35.8 | 2.80 | 357.7 | 238.64 | 1257.60 |
| ArcFace-iResNet100 | 65.12M | 12.098 | 241.96 | 4.1 | 24.20 | 41.3 | 789.49 | 4602.49 |
| MagFace-iResNet100 | 65.16M | 12.117 | 242.34 | 4.1 | 24.23 | 41.3 | 893.15 | 4800.20 |


The normalized estimates show that lightweight palmprint-specific models, such as CompNet, PPNet, and PalmBridge, require substantially less computation than the larger ArcFace- and MagFace-based models. However, GFLOPs alone cannot accurately predict actual smartphone latency because mobile performance also depends on the processor architecture, memory bandwidth, inference runtime, numerical precision, operator implementation, hardware delegate, and thermal conditions.
We will conduct an additional benchmark on two physical Android smartphones representing a budget/low-end device and a recent flagship device. We will update the rebuttal with the measured latency and recognition throughput once these experiments are completed during the rebuttal period.


>**2: Including more experimental analysis on performance drop reasons**

We thank the reviewer for this suggestion. We analyze the specific causes of performance degradation in Section 3.3. We will include a comprehensive analysis in the appendix to provide more details. 

The performance drop on the X-Palm dataset is due to significant domain shifts among sub-domains of this dataset. Specifically:
*   Cross-setting domain shift between scanner and smartphone settings significantly degrades the performance as the palm images are capture in totally different conditions (sensor, lighting, etc). 
*   Rolled and pitched captures drastically alter the visible palm geometry and partially occlude discriminative line patterns.
*   Wet surfaces blur and distort fine-grained creases and wrinkles due to reflections.
*   Handwritten text directly occludes the biometric palm surface.
*   Palm images of the Random domain are captured in uncontrolled conditions and may have two or more challenges (e.g., poor lighting, occlusion, and far distance) simultaneously, 

We evaluate the distributional (domain) gap between capture conditions in X-Palm dataset and compare it to baseline datasets. For every image, we extract a deep embedding using a pretrained vision backbone. We report results using DINOv2 ViT-S/14 below; the full analysis was independently repeated using an ImageNet-supervised ResNet-50 [1] as a cross-check, and the pattern of results was highly consistent across both backbones. For each dataset, we enumerate every pair of its own sub-domains (distinct acquisition sensors, illuminations, or environmental conditions) and compute five complementary distributional-shift metrics below between each pair, reporting the mean and standard deviation across all pairs. 

**Maximum Mean Discrepancy (MMD).** A kernel two-sample test statistic [2] that measures the distance between the mean embeddings of two distributions in a reproducing kernel Hilbert space. We use an RBF kernel with a data-adaptive (median-heuristic) bandwidth and the unbiased estimator. MMD is non-negative, with 0 indicating no detectable distributional difference between the two sub-domains.

**Proxy A-Distance (PAD).** PAD [3] estimates domain divergence as $2(1-2\epsilon)$, where $\epsilon$ is the cross-validated generalization error of a linear classifier trained to distinguish samples drawn from the two sub-domains. PAD ranges from 0 (the two sub-domains are indistinguishable to the classifier) to 2 (perfectly separable).

**Fréchet Feature Distance (FFD).** Fréchet Inception Distance [4] models each sub-domain's embeddings as a multivariate Gaussian. FFD is the closed-form Fréchet distance between the two Gaussians, combining a mean-shift term and a covariance-mismatch term into a single non-negative score.

**Kernel Inception Distance (KID).** An MMD-based alternative to FFD [5], using a polynomial rather than an RBF kernel. FFD's Gaussian-covariance estimate requires substantially more samples per sub-domain than we have to be reliable, and is known to be biased upward at small sample sizes; KID's kernel-based estimator was designed specifically to avoid this bias.

**Sliced Wasserstein Distance (SWD).** Approximates the Wasserstein distance between two distributions by averaging the closed-form 1-D Wasserstein distance over many random projections [6]. Unlike FFD, SWD makes no assumption that sub-domains are Gaussian-distributed. 

**Table 2: Within-dataset domain shift: mean $\pm$ standard deviation of each metric across all pairs of a dataset's own sub-domains (DINOv2 ViT-S/14 features).**

| **Dataset** | **# Sub-Dom.** | **Pairs** | **MMD ($\uparrow$)** | **PAD ($\uparrow$)** | **FFD ($\uparrow$)** | **KID ($\uparrow$)** | **SWD ($\uparrow$)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CASIA-MS | 6  | 15  | $0.181 \pm 0.081$ | $1.841 \pm 0.204$ | $210.4 \pm 117.7$ | $119.5 \pm 105.9$ | $1.188 \pm 0.377$ |
| MPDv2    | 2  | 1   | $0.010 \pm 0.000$ | $1.201 \pm 0.000$ | $14.2 \pm 0.0$    | $4.6 \pm 0.0$     | $0.284 \pm 0.000$ |
| X-Palm   | 17 | 136 | $\mathbf{0.373 \pm 0.268}$ | $1.796 \pm 0.299$ | $\mathbf{472.9 \pm 380.9}$ | $\mathbf{379.8 \pm 371.1}$ | $\mathbf{1.765 \pm 0.956}$ |
| XJTU-UP  | 4  | 6   | $0.151 \pm 0.053$ | $\mathbf{1.958 \pm 0.055}$ | $91.4 \pm 31.6$   | $23.6 \pm 8.5$    | $0.847 \pm 0.164$ |

X-Palm shows the largest mean pairwise MMD (0.373) and FFD (472.9), more than twice the next-highest dataset (XJTU-UP, 91.4). KID and SWD, included specifically to test whether FFD's small-sample bias or Gaussian assumption were driving this result, reproduce the identical ranking (KID: $379.8 > 119.5 > 23.6 > 4.6$; SWD: $1.77 > 1.19 > 0.85 > 0.28$, for X-Palm, CASIA-MS, XJTU-UP, and MPDv2 respectively).
While XJTU-UP and CASIA-MS both score higher PAD than X-Palm.

>**3: Distribution Analysis between Source and Target Domains**

In the previous section, we analyzed domain shifts among the X-palm's sub-domains and compared to baseline datasets. We also further analyze the distribution characteristics between the source and target domain using quantitative feature-space distribution analysis (e.g., t-SNE visualizations) and include the results in the revised version.

>**4: Domain Generalization and Domain Adaptation Methods**

We appreciate your suggestion. Our contribution is not developing a new method to decrease the domain gap between source and target domains, but novel benchmarking and domain-shift analysis frameworks, evaluating existing methods under these settings. We hope these findings will inspire future work on methods designed for this challenge.
As described in Section 3.1, we evaluated TSCAN (a Domain Adaptation method) and GIFT (a Domain Generalization method) specifically designed for the palmprint literature. Their cross-domain performances are reported in Tables 4 and 5 of the manuscript, demonstrating that while they offer some resilience, the compound variability of X-Palm still causes significant performance degradation.
We have used these baselines only in the closed-set cross-domain setting (not in the cross-dataset and open-set cross-domain settings) as these methods are originally presented and evaluated for closed-set scenarios.  We will include more DA and DG baselines in the appendix of the revised version.

>**5: References**

[1] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.‏

[2] Gretton, Arthur, et al. "A kernel two-sample test." The journal of machine learning research 13.1 (2012): 723-773.‏

[3] Ben-David, Shai, et al. "A theory of learning from different domains." Machine learning 79.1 (2010): 151-175.‏

[4] Heusel, Martin, et al. "Gans trained by a two time-scale update rule converge to a local nash equilibrium." Advances in neural information processing systems 30 (2017).‏

[5] Bińkowski, Mikołaj, et al. "Demystifying mmd gans." arXiv preprint arXiv:1801.01401 (2018).‏

[6] Rabin, Julien, et al. "Wasserstein barycenter and its application to texture mixing." International conference on scale space and variational methods in computer vision. Berlin, Heidelberg: Springer Berlin Heidelberg, 2011.‏






### reviewer #2 
We thank the reviewer for taking the time to write such a detailed review and for analyzing the paper in-depth. 
Below, we address each concern point-by-point and describe the manuscript changes we will make in the camera-ready version.

>**1: Dataset Limitations**

We appreciate the reviewer's advice and thank them for raising these points. We acknowledge the limitations of the current version of the X-Palm dataset, including its scale and imbalanced acquisition. We note that most existing datasets, such as CASIA-MS and XJTU-UP, are of comparable scale. 

| Dataset | Hands | Images | Devices | Age range | Paired scanner–smartphone data |
|---|---:|---:|---:|---:|:---:|
| CASIA-MS | 200 | 7,200 | 1 | 20–63 | No |
| XJTU-UP | 200 | 8,000 | 2 | 19–35 | No |
| MPD-v2 | 400 | 16,000 | 2 | 20–50 | No |
| **X-Palm** | **206** | **6,006** | **80+** | **18–76** | **Yes** |

Scaling up participant recruitment is challenging due to the time-consuming curation process, privacy considerations, and IRB ethical constraints. Under our IRB-approved protocol, biometric acquisition was restricted to approved private indoor locations to protect participant privacy and prevent observation by unauthorized individuals.

The custom multispectral scanner was also installed at a fixed institutional location and could not be relocated. These requirements geographically constrained recruitment and contributed to the demographic imbalance and imbalance between the paired and smartphone-only subsets. Within these constraints, we made deliberate efforts to recruit a diverse group of participants, and the resulting demographic distribution broadly reflects that of the local community. However, because participant demographics remain imbalanced, we chose not to define or split domains based on race, gender, or other demographic attributes.
X-Palm includes 40.78% female participants and 11 self-reported ethnic groups. Nevertheless, we acknowledge that the resulting demographic distribution remains imbalanced, particularly toward younger participants and demographic groups that were more accessible at the approved collection locations. We therefore do not claim that the current release establishes demographic fairness or population-level robustness.

The main contribution of X-Palm is not the population scale, but its **cross-domain design**. It captures the same participant data under controlled multispectral enrollment and unconstrained, participant-driven smartphone authentication, while incorporating variations in device, illumination, pose, perspective, distance, background, occlusion, and palm-surface condition. This enables systematic study of the practical deployment gap between controlled enrollment and smartphone authentication, which is not directly supported by existing datasets of comparable identity scale.

Regarding privacy risks and gated-access reproducibility, we have stored the anonymized dataset in our secured institutional repository to protect participant privacy. Reproducibility is nonetheless ensured: our experiments are fully reproducible using the open-source code we provide, and access to X-Palm and the other datasets can be requested through the standard access procedure. 

We will revise the limitation section to explicitly acknowledge the limited identity count, the acquisition and demographic imbalance, and the resulting limits on fairness and generalization claims.


>**2: Ethical/Privacy Documentations needs to be more directly verifiable**

To safeguard participant privacy and mitigate re-identification risk, the raw data is stored on an access-controlled and encrypted in our institutional repository. Access is strictly gated: prospective researchers must digitally sign an End User License Agreement (EULA) that explicitly prohibits any re-identification attempts or commercial exploitation before receiving a secure, time-limited download link.

>**3: Effect of test set size on performance confidence in open-set cross-domain setting**

We thank the reviewer for this invaluable comment. We employ Cross-Validation (CV) and Confidence-Interval (CI) for this evaluation. Both CV and CI results show the same underlying pattern when the effective train/test ratio shrinks the test partition: comparing 3-fold (test $\approx 33\%$ per fold) to 5-fold (test $= 20\%$ per fold), the standard deviation increases for both EER and Rank-1 across the 12 test domains; correspondingly, comparing the ratio=0.5 bootstrap to ratio=0.8, the mean 95% CI half-width widens for both EER and Rank-1. This consistent direction across two independent evaluation protocols confirms that uncertainty scales with the size of the held-out identity set. We will include both CV and CI results in the revised manuscript and release the exact fold partitions as a fixed, official identity split, so that future work can make comparisons under identical train/test identity assignments rather than ad hoc re-splits. We include the same analysis for all baselines, also for the cross-dataset and closed-set cross-domain setting in the appendix section of the revised version.

**Table 1: CompNet performance stability under 3-fold vs. 5-fold identity-disjoint cross-validation, all 12 test domains.**

| **Test Domain** | **3-Fold EER_mean** | **3-Fold EER_std** | **3-Fold Rank1_mean** | **3-Fold Rank1_std** | **5-Fold EER_mean** | **5-Fold EER_std** | **5-Fold Rank1_mean** | **5-Fold Rank1_std** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Scanner | 29.057 | 0.888 | 73.56 | 2.62 | 28.988 | 1.263 | 79.23 | 4.06 |
| Smartphone | 22.549 | 0.475 | 65.60 | 2.85 | 22.392 | 0.900 | 72.81 | 2.59 |
| Wet & Text | 19.696 | 2.140 | 55.84 | 2.26 | 14.983 | 3.512 | 61.17 | 4.28 |
| Wet & RND | 23.327 | 0.338 | 37.78 | 3.15 | 23.755 | 3.184 | 45.68 | 5.62 |
| RND & Text | 24.060 | 2.693 | 34.74 | 1.32 | 22.745 | 2.292 | 43.16 | 5.42 |
| SF & Roll | 26.893 | 0.185 | 25.70 | 0.71 | 30.889 | 5.970 | 35.16 | 6.78 |
| JF & Pitch | 24.492 | 3.462 | 41.96 | 8.90 | 22.161 | 3.671 | 48.38 | 8.06 |
| BF & Far | 13.364 | 0.350 | 59.33 | 5.67 | 13.052 | 4.586 | 73.29 | 5.26 |
| Roll & Close | 25.132 | 2.201 | 39.67 | 7.52 | 24.984 | 3.859 | 41.92 | 7.36 |
| Far & JF | 14.327 | 1.246 | 55.27 | 3.58 | 14.694 | 2.648 | 67.89 | 7.70 |
| FO & SF | 8.373 | 0.814 | 79.48 | 2.17 | 8.108 | 2.087 | 85.26 | 5.16 |
| Roll & Pitch | 29.223 | 2.733 | 26.55 | 3.48 | 29.214 | 3.307 | 33.32 | 3.12 |

**Table 2: CompNet bootstrap 95% CI (identity-level resampling, $B=1000$) under ratio=0.5 vs. ratio=0.8 train/test splits, all 12 test domains.**

| **Test Domain** | **Ratio = 0.5 EER % [95% CI]** | **Ratio = 0.5 Rank-1 % [95% CI]** | **Ratio = 0.8 EER % [95% CI]** | **Ratio = 0.8 Rank-1 % [95% CI]** |
| :--- | :--- | :--- | :--- | :--- |
| Scanner | 29.76 [27.73, 31.72] | 68.15 [65.55, 74.39] | 27.99 [25.34, 31.15] | 74.67 [70.23, 85.12] |
| Smartphone | 22.17 [18.55, 26.22] | 73.31 [70.77, 82.52] | 22.32 [18.45, 26.50] | 73.01 [69.90, 83.08] |
| Wet & Text | 21.05 [13.51, 28.44] | 44.21 [36.84, 58.95] | 13.16 [ 4.46, 19.87] | 68.42 [55.26, 86.84] |
| Wet & RND | 28.43 [22.01, 35.64] | 26.32 [21.05, 38.97] | 18.42 [ 8.57, 36.05] | 52.63 [36.84, 71.05] |
| RND & Text | 28.42 [20.61, 37.01] | 26.32 [20.00, 38.95] | 25.75 [13.51, 38.16] | 44.74 [31.58, 63.16] |
| SF & Roll | 31.58 [24.88, 39.75] | 17.89 [12.63, 30.53] | 28.95 [16.21, 44.05] | 28.95 [18.42, 50.00] |
| JF & Pitch | 25.26 [18.46, 33.83] | 31.58 [24.21, 43.16] | 28.95 [16.67, 43.42] | 42.11 [26.32, 57.89] |
| BF & Far | 15.82 [10.66, 23.98] | 55.79 [50.53, 69.47] | 12.02 [ 6.42, 30.82] | 68.42 [55.26, 86.84] |
| Roll & Close | 29.03 [22.37, 36.03] | 26.32 [20.00, 38.95] | 23.68 [11.76, 35.72] | 42.11 [31.58, 63.16] |
| Far & JF | 13.68 [ 8.73, 18.82] | 48.42 [42.11, 62.11] | 18.42 [ 9.18, 29.02] | 52.63 [39.47, 73.68] |
| FO & SF | 9.47 [ 5.35, 15.56] | 70.53 [65.26, 83.16] | 9.53 [ 2.22, 20.52] | 84.21 [73.68, 97.37] |
| Roll & Pitch | 30.61 [21.20, 40.32] | 21.05 [15.79, 34.74] | 23.68 [12.49, 38.50] | 39.47 [26.32, 63.16] |

>**4: ROI Extraction Pipeline**

Quality control details will be provided in Appendix A.5. We use a custom-built annotation tool (Figure 5) where an operator (the author) marks anatomical keypoints that unambiguously determine the palm RoI. All annotations are performed by the first author, and the tool provides a live visual preview of the extracted square ROI, allowing immediate quality control. Low-quality images (e.g., blurry and full palm occlusion) are discarded to guarantee the overall quality if the dataset. 

To evaluate the annotation consistency of the only annotator (the author) over time, we repeated the annotation of 10 palm images for 10 iterations. The extracted RoIs demonstrated high spatial consistency over iterations.  

We manually performed RoI extraction to provide reliable data for benchmarking, eliminating the effect of ROI extraction errors. We also enrich the X-Palm dataset with paired raw images and ground-truth extracted ROIs that can be used for development of the automatic ROI extraction pipelines. Notably, X-Palm dataset is the first dataset that provides paired raw palm images and ground-truth RoIs along with other useful metadata.  

>**5: What users receive?**

Unlike previous palmprint datasets that provide only raw palm images (e.g., CASIA-MS, MPD-v2) or only extracted RoIs (e.g., XJTU-UP), we provide the raw palm images, extracted ROIs, the keypoint coordinates, and anonymized metadata including age group, ethnicity group, gender, and smartphone model used for image capturing, and the official split files used for experiments (to enable reproducibility), upon request under the strict EULA to prevent misuse.

**Package Contents:**
- `LICENSE_EULA.md`: non-commercial research EULA for approved users.
- `CITATION.cff`: machine-readable citation metadata.
- `xpalm_croissant.json`: machine-readable Croissant metadata for ML dataset discovery and responsible AI documentation.
- `data_split.json`: official splits of the datasets in all experiments for reproducibility. 
- `metadata/file_manifest.csv`: file-level metadata for all released images.
- `metadata/participant_metadata.csv`: anonymous subject-level coverage metadata.
- `metadata/data_dictionary.csv`: descriptions of metadata fields.
- `data/`: restricted-access image data, organized by modality (scanner/smartphone) and processing stage (raw images / extracted RoIs), and also key point coordinates for RoI extraction.




### reviewer #3

We thank the reviewer for the detailed and helpful feedback. Below we respond to each concern.

>**1. Dataset Limitation**

We thank the reviewer for recognizing the novelty of our paired design. We acknowledge the current dataset's limited scale and imbalanced acquisition. Further recruitment is constrained by the time-consuming curation process, privacy considerations, and ethical constraints; we tried to collect data at least at the scale of most existing datasets (CASIA-MS, XJTU-UP). Our scanner was fixed in place and could not be relocated, geographically constraining recruitment and causing imbalance across devices (smartphone vs. scanner) and ethnic groups. We recruited as diverse a group as these constraints allowed, and demographics broadly reflect the local community; since they remain imbalanced, we did not split domains by race, gender, or other demographic attributes. (we could not provide more details due to limited space, please check "Dataset Limitations" of our answer for the reviewer XTTj for more details)

>**2. Imbalance Effect on Training and Evaluation**

We ran two controlled experiments to assess the effect of scanner/smartphone acquisition imbalance.

**Training-time imbalance** (Table 1): equal-size training sets on the same test set, Mode A (all dual-domain) vs. Mode B (a third replaced with smartphone-only identities). Mode B has higher EER and lower Rank-1 due to the imbalanced training set. 

**Table 1: Effect of domain imbalance on training. with 95% CI**

| **Mode** | **EER (%)** | **Rank-1 (%)** |
| :--- | :--- | :--- |
| Mode A (balanced: all dual-domain) | 16.15 [13.47, 19.18] | 84.56 [82.94, 91.42] |
| Mode B (imbalanced: dual + smartphone-only) | 17.90 [15.34, 21.06] | 79.87 [78.47, 87.94] |


**Inference-time imbalance** (Table 2): same trained model/probe set, only gallery composition varies: Mode 1 (mixed-domain gallery) vs. Mode 2 (smartphone-only gallery). EERs are close with overlapping CIs, but Rank-1 drops 19 points with disjoint CIs (Mode 1: [77.04, 88.18] vs. Mode 2: [62.70, 76.58]), a smartphone-only gallery hurts Rank-1 though EER separability is largely unaffected.

**Table 2: Effect of domain imbalance on inference. with 95% CI.**

| **Mode** | **EER (%)** | **Rank-1 (%)** |
| :--- | :--- | :--- |
| Mode 1 (mixed-domain gallery) | 17.0 [14.248, 19.856] | 78.80 [77.04, 88.18] |
| Mode 2 (smartphone-only gallery) | 16.51 [13.509, 19.409] | 63.29 [62.70, 76.58] |

We will add further baselines and imbalance ratios in the appendix.

>**3. RoI Extraction Pipeline**

Quality-control details will appear in Appendix A.5. Annotation uses a custom tool with a live preview (please see Figure 5) enabling immediate QC; low-quality images (blurry, occluded) are discarded. A repeatability check, re-annotating 10 images over 10 iterations, showed high spatial consistency. RoIs were extracted manually to eliminate extraction-error effects on benchmarking. X-Palm also provides paired raw images and ground-truth RoIs plus metadata, to our knowledge the first dataset to do so, supporting future automatic RoI-extraction research.

>**4. Low Quality Images or Significant Domain Shifts?**

We appreciate this important question that need to be clarified. We address this with two analyses: image quality and domain shift, across CASIA-MS, MPDv2, XJTU-UP, and both X-Palm subsets (scanner, smartphone).

### Image Quality Analysis
All images are resized to $112\times112$ (the recognizer's input size) before computing metrics, so comparisons reflect quality as the recognizer sees it. Metrics used:

- **Sharpness**: Laplacian variance [7] and Tenengrad [8] (Sobel-gradient-based), both lower for blurrier images.
- **Contrast/information**: RMS contrast [9] (intensity std. dev.) and Shannon entropy [10] (histogram information content); both lower for low-detail images.
- **Noise proxy (Pseudo-PSNR)**: compares each image to its own Gaussian-blurred version rather than requiring a pristine reference as in standard PSNR [11]; higher means less relative high-frequency noise.
- **Gabor-based ridge energy**: mean response of a multi-orientation Gabor filter bank, relevant since palmprint recognition often relies on Gabor-based texture coding (e.g., PalmCode [12]).

**Table 3: Image quality metrics across datasets, computed at a fixed $112\times112$ evaluation resolution.**

| **Metric** | **XJTU-UP** | **MPDv2** | **CASIA-MS** | **X-Palm (Scanner)** | **X-Palm (Smartphone)** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Laplacian Var. ($\uparrow$) | $43.04 \pm 34.94$ | $66.88 \pm 39.73$ | $7.71 \pm 6.84$ | $18.93 \pm 33.03$ | $152.55 \pm 190.08$ |
| Tenengrad ($\uparrow$) | $388.36 \pm 230.88$ | $631.95 \pm 465.71$ | $182.28 \pm 146.46$ | $572.80 \pm 554.22$ | $1296.72 \pm 1118.27$ |
| RMS Contrast ($\uparrow$) | $8.70 \pm 3.18$ | $13.17 \pm 4.81$ | $9.18 \pm 2.74$ | $11.00 \pm 5.59$ | $16.85 \pm 7.80$ |
| Entropy ($\uparrow$) | $5.00 \pm 0.46$ | $5.56 \pm 0.44$ | $5.09 \pm 0.38$ | $5.16 \pm 0.59$ | $5.82 \pm 0.57$ |
| Pseudo-PSNR ($\uparrow$) | $44.59 \pm 3.19$ | $42.26 \pm 2.16$ | $49.81 \pm 2.36$ | $47.37 \pm 4.28$ | $39.75 \pm 3.52$ |
| Ridge Energy ($\uparrow$) | $2.66 \pm 0.29$ | $1.87 \pm 0.25$ | $1.36 \pm 0.22$ | $1.89 \pm 0.61$ | $2.30 \pm 0.35$ |

**Scanner vs. CASIA-MS:** Scanner subset scores higher on most metrics (Laplacian, Tenengrad, RMS contrast, entropy, ridge energy); CASIA-MS only wins on Pseudo-PSNR. So the scanner subset is not lower quality than CASIA-MS; it's sharper and richer in ridge texture.

**Smartphone vs. XJTU-UP/MPDv2:** Smartphone subset scores higher on Laplacian variance, Tenengrad, RMS contrast, entropy (both comparisons), and ridge energy vs. MPDv2. Against XJTU-UP, ridge energy is lower (2.30 vs. 2.66), meaning XJTU-UP retains clearer palm-line structure despite lower generic sharpness/contrast. Overall, the smartphone subset is not lower quality by conventional standards.

### Domain Shift Analysis
As a complementary explanation, we measure distributional gaps between capture conditions using deep embeddings (DINOv2 ViT-S/14 below; results were highly consistent when repeated with ImageNet-supervised ResNet-50 [1]). For each dataset, we compute distributional-shift metrics between every pair of its own sub-domains (sensors/illumination/conditions):

- **MMD** [2]: kernel two-sample test distance between mean embeddings (RBF kernel, median-heuristic bandwidth); 0 = no detectable difference.
- **PAD** [3]: $2(1-2\epsilon)$ where $\epsilon$ is a linear classifier's error distinguishing the two sub-domains; ranges 0 (indistinguishable) to 2 (perfectly separable).
- **FFD** [4]: Fréchet distance between Gaussian fits of each sub-domain's embeddings (mean-shift + covariance-mismatch terms).
- **KID** [5]: MMD-based alternative to FFD using a polynomial kernel; avoids FFD's upward bias at small sample sizes.
- **SWD** [6]: averages closed-form 1-D Wasserstein distance over many random projections; unlike FFD, assumes no Gaussianity.

**Table 4: Within-dataset domain shift: mean $\pm$ standard deviation of each metric across all pairs of a dataset's own sub-domains (DINOv2 ViT-S/14 features).**

| **Dataset** | **# Sub-Dom.** | **Pairs** | **MMD ($\uparrow$)** | **PAD ($\uparrow$)** | **FFD ($\uparrow$)** | **KID ($\uparrow$)** | **SWD ($\uparrow$)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CASIA-MS | 6  | 15  | $0.181 \pm 0.081$ | $1.841 \pm 0.204$ | $210.4 \pm 117.7$ | $119.5 \pm 105.9$ | $1.188 \pm 0.377$ |
| MPDv2    | 2  | 1   | $0.010 \pm 0.000$ | $1.201 \pm 0.000$ | $14.2 \pm 0.0$    | $4.6 \pm 0.0$     | $0.284 \pm 0.000$ |
| X-Palm   | 17 | 136 | $\mathbf{0.373 \pm 0.268}$ | $1.796 \pm 0.299$ | $\mathbf{472.9 \pm 380.9}$ | $\mathbf{379.8 \pm 371.1}$ | $\mathbf{1.765 \pm 0.956}$ |
| XJTU-UP  | 4  | 6   | $0.151 \pm 0.053$ | $\mathbf{1.958 \pm 0.055}$ | $91.4 \pm 31.6$   | $23.6 \pm 8.5$    | $0.847 \pm 0.164$ |

X-Palm has the largest mean pairwise MMD (0.373) and FFD (472.9), over twice the next-highest (XJTU-UP: 91.4). KID and SWD, included to check whether FFD's small-sample bias or Gaussian assumption drove this, reproduce the same ranking, though XJTU-UP and CASIA-MS both score higher on PAD than X-Palm.

Together with the image quality results and cross-dataset evaluation, this shows X-Palm's performance drop is not attributable to lower image quality, its score higher than the other datasets on most quality and sub-domain-distance metrics. The drop instead reflects genuine domain shift from real-world acquisition variation.

>**5. References**

[1] He, Kaiming, et al. "Deep residual learning for image recognition." CVPR. 2016.

[2] Gretton, Arthur, et al. "A kernel two-sample test." JMLR 13.1 (2012): 723-773.

[3] Ben-David, Shai, et al. "A theory of learning from different domains." Machine learning 79.1 (2010): 151-175.

[4] Heusel, Martin, et al. "GANs trained by a two time-scale update rule converge to a local Nash equilibrium." NeurIPS 30 (2017).

[5] Bińkowski, Mikołaj, et al. "Demystifying MMD GANs." arXiv:1801.01401 (2018).

[6] Rabin, Julien, et al. "Wasserstein barycenter and its application to texture mixing." Scale Space and Variational Methods in Computer Vision. Springer, 2011.

[7] Pech-Pacheco, José Luis, et al. "Diatom autofocusing in brightfield microscopy: a comparative study." ICPR. IEEE, 2000.

[8] Santos, Andrés, et al. "Evaluation of autofocus functions in molecular cytogenetic analysis." Journal of Microscopy 188.3 (1997): 264-272.

[9] Peli, Eli. "Contrast in complex images." JOSA A 7.10 (1990): 2032-2040.

[10] Shannon, Claude Elwood. "A mathematical theory of communications." Bell System Technical Journal 27 (1948): 379-423.

[11] Huynh-Thu, Quan, and Mohammed Ghanbari. "Scope of validity of PSNR in image/video quality assessment." Electronics Letters 44.13 (2008): 800-801.

[12] Zhang, David, et al. "Online palmprint identification." IEEE TPAMI 25.9 (2003): 1041-1050.
