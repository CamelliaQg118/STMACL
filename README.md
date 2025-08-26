# STMACL (10.1016/j.eswa.2025.129426)
An official source code for the paper "Integrating modularity maximization and contrastive learning for identifying spatial domain from spatial transcriptomics," accepted by Expert Systems with Applications 2025 (https://doi.org/10.1016/j.eswa.2025.129426). Any communications or issues are welcome. Please contact qigao118@163.com. If you find this repository useful to your research or work, it is really appreciated to star this repository. ❤️
## Overview:
![Figure_1](https://github.com/user-attachments/assets/b19d0837-92dd-49e6-87f1-06420cf6eb77)


STMACL, a self-supervised learning framework, combines modularity maximization and contrastive learning for spatial clustering. STMACL has been applied to seven spatial transcriptomics datasets across platforms like omsFISH, STARmap,10X Visium, Stereo-seq, and Slide-seqV2, proving its capability to deliver enhanced representations for a range of downstream analyses, such as clustering, visualization, trajectory inference, and differential gene analysis.

__STMACL__
## Requirements:
 
STMACL is implemented in the pytorch framework. Please run STMACL on CUDA. The following packages are required to be able to run everything in this repository (included are the versions we used):

```bash
python==3.10.0
torch==2.1.0
cudnn==11.8
numpy==1.25.2
scanpy==1.11.0
pandas==2.2.3
scipy==1.15.2
scikit-learn==1.5.2
anndata==0.11.3
R==4.3.3
ryp2==3.5.11
tqdm==4.67.1
matplotlib==3.10.1
seaborn==0.13.2
```
