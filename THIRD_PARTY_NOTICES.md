# Third-Party Notices

This repository depends on or interoperates with several third-party components
released under their own licenses. The following list summarizes those
components and the obligations they carry. Please consult each project's
upstream repository for the authoritative license text.

## Pretrained models

### MS-CLAP (Microsoft CLAP)
- Source: https://github.com/microsoft/CLAP
- License: Microsoft Research License (research/non-commercial use)
- Citation: Elizalde et al., "CLAP: Learning Audio Concepts From Natural Language Supervision", 2023.

### LAION-CLAP
- Source: https://github.com/LAION-AI/CLAP
- License: CC0-1.0
- Citation: Wu et al., "Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation", 2023.

### WavLM
- Source: https://huggingface.co/microsoft/wavlm-base-plus
- License: MIT
- Citation: Chen et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing", IEEE J-STSP 2022.

### wav2vec 2.0 / XLSR
- Source: https://github.com/facebookresearch/fairseq
- License: MIT
- Citation: Conneau et al., "Unsupervised Cross-lingual Representation Learning for Speech Recognition", Interspeech 2021.

## Baselines

### AASIST / AASIST-L
- Source: https://github.com/clovaai/aasist
- License: MIT
- Citation: Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks", ICASSP 2022.
- Notes: Fetched on demand by `setup_baselines.sh`. Not redistributed in this repository.

### RawNet2
- Source: https://github.com/asvspoof-challenge/2021
- License: MIT
- Citation: Tak et al., "End-to-end anti-spoofing with RawNet2", ICASSP 2021.

## Datasets

This repository does not redistribute any dataset. Users are required to
obtain datasets from their original sources and accept the relevant terms.

| Dataset | Source | License / Terms |
|---|---|---|
| LibriSpeech | https://www.openslr.org/12/ | CC-BY-4.0 |
| ASVspoof 2019 LA | https://www.asvspoof.org/ | ODC-BY (research use) |
| ASVspoof 5 | https://www.asvspoof.org/ | ODC-BY (research use) |
| In-the-Wild | https://deepfake-total.com/in_the_wild | Research use, free download |
| MUSDB18 | https://sigsep.github.io/datasets/musdb.html | Research only (no commercial use) |
| FakeMusicCaps | https://huggingface.co/datasets/fakemusiccaps | CC-BY-NC (verify upstream) |
| FMA | https://github.com/mdeff/fma | CC-BY |
| SONICS | https://huggingface.co/datasets/SONICS-LMM/SONICS | Research use |
| MLAAD | https://owncloud.fraunhofer.de/index.php/s/tL2Y1FKrWiX4ZtP | Research use |
| AUDETER | https://huggingface.co/datasets/audeter | Research use |
| DeepSpeak v2 | https://huggingface.co/datasets/faridlab/deepspeak_v2 | Research use |
| FakeAVCeleb | https://github.com/DASH-Lab/FakeAVCeleb | Research agreement required |
| SpoofCeleb | https://huggingface.co/datasets/jungjee/SpoofCeleb | Research use |

## Python dependencies

Direct dependencies are listed in `requirements.txt`. Each package is
licensed under its own terms (most are MIT or BSD); please refer to the
respective project for license text.
