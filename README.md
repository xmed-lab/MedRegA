# MedRegA: Interpretable Bilingual Multimodal Large Language Model for Diverse Biomedical Tasks

<a href="https://arxiv.org/abs/2410.18387"><img src="https://img.shields.io/badge/Paper-arxiv-green.svg?style=flat-square"></a>
<a href="https://medrega.github.io/"><img src="https://img.shields.io/badge/Project-Website-blue.svg?style=flat-square"></a>
<a href="https://huggingface.co/Luxuriant16/medrega"><img src="https://img.shields.io/badge/Model-HuggingFace-red.svg?style=flat-square"></a>
<a href="https://huggingface.co/datasets/Luxuriant16/MedRegInstruct"><img src="https://img.shields.io/badge/Dataset-HuggingFace-orange.svg?style=flat-square"></a>

**MedRegA**, an interpretable bilingual generalist model for diverse biomedical tasks, represented by its outstanding ability to leverage regional information. MedRegA can perceive 8 modalities covering almost all the body parts, showcasing significant versatility.

<img src="asset\intro.png" width=70% >

## Overview

💡We establish **Region-Centric tasks** with a large-scale dataset, **MedRegInstruct**, where each sample is paired with coordinates of body structures or lesions.

💡Based on the proposed dataset, we develop a **Region-Aware medical MLLM**, **MedRegA**, as a bilingual generalist medical AI system to perform both image-level and region-level medical vision-language tasks, demonstrating impressive versatility. 

## Schedule

+ [x] Release the model.
+ [x] Release the demo code.
+ [x] Release the evaluation code.
+ [x] Release the training code.
+ [x] Release the data.

## Environment

Please refer to [InternVL Installation](https://internvl.readthedocs.io/en/latest/get_started/installation.html) to build the environment.

## Demo

Run the demo:

```bash
torchrun --nproc-per-node=1 src/demo.py
```

## Training

The training details can be referred to the [slurm scripts](https://github.com/xmed-lab/MedRegA/tree/main/src/shell/internvl_chat_v1_2_hermes2_yi34b_448_finetune_continue_lora.sh) for multi-node multi-gpu training. Before training, the dataset should be organized as [MedRegInstruct](https://huggingface.co/datasets/Luxuriant16/MedRegInstruct) and registered in [`meta_file.json`](https://github.com/xmed-lab/MedRegA/tree/main/src/shell/meta_file.json).

## Cite

```
@article{wang2024interpretable,
  title={Interpretable bilingual multimodal large language model for diverse biomedical tasks},
  author={Wang, Lehan and Wang, Haonan and Yang, Honglong and Mao, Jiaji and Yang, Zehong and Shen, Jun and Li, Xiaomeng},
  journal={arXiv preprint arXiv:2410.18387},
  year={2024}
}
```

## Acknowledgement

We refer to the codes from [InternVL](https://github.com/OpenGVLab/InternVL). Thank the authors for releasing their code.
