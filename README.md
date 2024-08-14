# KNN-MT-DR
[![GitHub license](https://badgen.net/github/license/Naereen/Strapdown.js)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE) [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Active](http://img.shields.io/badge/Status-Active-green.svg)](https://tterb.github.io) [![Ask Me Anything !](https://img.shields.io/badge/Ask%20me-anything-1abc9c.svg)](https://GitHub.com/Naereen/ama)

Code for "Efficient k-Nearest-Neighbor Machine Translation with Dynamic Retrieval" (Findings of ACL 2024).

## Requirements and Installation
* python >= 3.7
* pytorch >= 1.10.0
* faiss-gpu >= 1.7.3
* sacremoses == 0.0.41
* sacrebleu == 1.5.1
* fastBPE == 0.1.0
* streamlit >= 1.13.0
* scikit-learn >= 1.0.2
* seaborn >= 0.12.1

You can install this toolkit by
```shell
git clone https://github.com/DeepLearnXMU/knn-mt-dr.git
cd knn-mt-dr
pip install --editable ./
```

Note: Installing faiss with pip is not suggested. For stability, we recommand you to install faiss with conda

```bash
CPU version only:
conda install faiss-cpu -c pytorch

GPU version:
conda install faiss-gpu -c pytorch # For CUDA
```

## Overview

You can prepare pretrained models and dataset by executing the following command:

```bash
cd knnbox-scripts
bash prepare_dataset_and_model.sh
cp ../pretrain-models/wmt19.de-en/dict.en.txt ../pretrain-models/wmt19.de-en/fairseq-vocab.txt
```

> use bash instead of sh. If you still have problem running the script, you can manually download the [wmt19 de-en single model](https://github.com/facebookresearch/fairseq/blob/main/examples/wmt19/README.md) and [multi-domain de-en dataset](https://drive.google.com/file/d/18TXCWzoKuxWKHAaCRgddd6Ub64klrVhV/view), and put them into correct directory (you can refer to the path in the script).

You can build datastore by executing the following command:

```bash
cd skip-knn-mt
python get_output_projection.py
bash build_datastore.sh
bash build_valid_datastore.sh
bash prepare_dataset.sh
```

You can train model and inference by executing the following command:

```bash
bash train.sh
bash skip_inference.sh
```

## Acknowledgement
[kNN-box](https://github.com/NJUNLP/knn-box): the codebase we built upon. This repository is an open-source toolkit to build kNN-MT models. We greatly appreciate the excellent foundation provided by the authors.

You can refer to [kNN-box](https://github.com/NJUNLP/knn-box) for more detailed information.

## Citation
If you found this repository helpful in your research, please consider citing:
```bibtex
@article{gao2024efficient,
  title={Efficient k-Nearest-Neighbor Machine Translation with Dynamic Retrieval},
  author={Gao, Yan and Cao, Zhiwei and Miao, Zhongjian and Yang, Baosong and Liu, Shiyu and Zhang, Min and Su, Jinsong},
  journal={arXiv preprint arXiv:2406.06073},
  year={2024}
}
```
