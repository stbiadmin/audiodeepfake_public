# Datasets

This directory holds the datasets used for training and evaluation. Datasets
are not redistributed in this repository; download each one from the original
source listed below and place it under `data/<dataset_name>/`.

The `data/sound-samples/` subdirectory ships with the repository and contains
a small set of demo clips used by the quick-start example and the smoke test.

## Speech datasets

### LibriSpeech (real)
- Source: <https://www.openslr.org/12/>
- License: CC-BY-4.0
- Citation: Panayotov et al., "Librispeech: an ASR corpus based on public
  domain audio books," ICASSP 2015.
- Helper: `python scripts/download_librispeech.py`
- Expected path: `data/librispeech/`

### ASVspoof 2019 LA (fake)
- Source: <https://www.asvspoof.org/index2019.html>
- License: ODC-BY (research use)
- Citation: Wang et al., "ASVspoof 2019: A large-scale public database of
  synthetized, converted and replayed speech," CSL 2020.
- Helper: `python scripts/download_asvspoof.py`
- Expected path: `data/asvspoof2019/LA/`

### DeepSpeak v2
- Source: <https://huggingface.co/datasets/faridlab/deepspeak_v2>
- License: Research use
- Helper: `python scripts/download_deepspeak.py`
- Expected path: `data/deepspeak_v2/`

### MLAAD
- Source: <https://owncloud.fraunhofer.de/index.php/s/tL2Y1FKrWiX4ZtP>
- License: Research use
- Helper: `python scripts/download_mlaad.py`
- Expected path: `data/mlaad/`

### AUDETER (modern TTS, 2024-2025)
- Source: <https://huggingface.co/datasets/audeter>
- License: Research use
- Expected path: `data/audeter/`

### In-the-Wild (evaluation)
- Source: <https://deepfake-total.com/in_the_wild>
- License: Research use, free download
- Expected path: `data/in_the_wild/`

### ASVspoof 5 (evaluation)
- Source: <https://www.asvspoof.org/>
- License: ODC-BY (research use)
- Expected path: `data/asvspoof5/`

### FakeAVCeleb / SpoofCeleb (optional eval sets)
- FakeAVCeleb: <https://github.com/DASH-Lab/FakeAVCeleb> (research agreement)
- SpoofCeleb: <https://huggingface.co/datasets/jungjee/SpoofCeleb>

## Music datasets

### MUSDB18 / MUSDB18-HQ (real)
- Source: <https://sigsep.github.io/datasets/musdb.html>
- License: Research use only
- Citation: Rafii et al., "MUSDB18 - a corpus for music separation," 2017.
- Helper: `python scripts/download_musdb18.py`
- Expected path: `data/musdb18_stems/`

### FakeMusicCaps (fake)
- Source: <https://huggingface.co/datasets/fakemusiccaps>
- License: CC-BY-NC (verify upstream)
- Helper: `python scripts/download_fakemusiccaps.py`
- Expected path: `data/fakemusiccaps/`

### FMA (real, cross-domain evaluation)
- Source: <https://github.com/mdeff/fma>
- License: CC-BY
- Citation: Defferrard et al., "FMA: A Dataset For Music Analysis," ISMIR 2017.
- Expected path: `data/fma/`

### SONICS (fake, cross-domain evaluation)
- Source: <https://huggingface.co/datasets/SONICS-LMM/SONICS>
- License: Research use
- Helper: `python scripts/download_sonics.py`
- Expected path: `data/sonics/`

## Suggested directory layout

```
data/
├── librispeech/
├── asvspoof2019/LA/
├── deepspeak_v2/
├── mlaad/
├── audeter/
├── musdb18_stems/
├── fakemusiccaps/
├── fma/
├── sonics/
├── in_the_wild/
├── asvspoof5/
└── sound-samples/         (shipped with the repository)
```

After download, run feature extraction once per dataset; see
`REPRODUCIBILITY.md` for the commands.
