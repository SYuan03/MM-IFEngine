# MM-IFEngine: Towards Multimodal Instruction Following
<div align="center">

[📃[Paper](https://arxiv.org/abs/2504.07957)]
[🌐[Project Page](https://syuan03.github.io/MM-IFEngine/)]
[🤗[Hugging Face](https://github.com/SYuan03/MM-IFEngine)]
[🛠️[Evaluation](https://github.com/SYuan03/MM-IFEngine)]
</div>

<div align="center">
 <img src="./webpages/images/teaser3_00.png" width="800"/>
</div>

## 📣 What's New
- **[2025.4.24]** [MMIFEval](https://github.com/open-compass/VLMEvalKit/pull/938) has been merged into [VLMEvalkit](https://github.com/OpenCompass/VLMEvalkit). You can directly evaluate your model on MMIFEval with it ! Usage see [Evaluation using VLMEvalkit](https://github.com/SYuan03/MM-IFEngine?tab=readme-ov-file#option-1-recommended-evaluation-using-vlmevalkit) or more on Official repo of [VLMEvalkit](https://github.com/OpenCompass/VLMEvalkit) ! 🎉🎉🎉
- **[2025.4.11]** Our MM-IFEngine Paper is released ! Check it at 📃[Arxiv: MM-IFEngine](https://arxiv.org/abs/2504.07957) ! Our Dataset will be open-sourced soon ! 🎉🎉🎉

## 🌟 Highlights
<div align="center">
 <img src="./webpages/images/pipeline_00.png" width="800"/>
</div>

1. An MM-IFEngine pipeline
for generating multimodal constraint-rich image-instruction
pairs; 
2. A large-scale training dataset MM-IFInstruct-23k
and preference optimization dataset MM-IFDPO-23k de-
rived from MM-IFEngine;
3. A challenging multimodal instruction following benchmark MM-IFEval with diverse constraints and comprehensive evaluation approaches; 
4. Empirical evidence showing significant performance gains on
both our MM-IFEval and existing benchmarks when training
MLLMs on MM-IFInstruct-23k via SFT and MM-IFDPO-23k via DPO.

## 📚 Dataset Statistics
<div align="center">
 <img src="./webpages/images/dataset-statistic.png" width="800"/>
</div>

## 🏆 MM-IFEval Leaderboard
<div align="center">
 <img src="./webpages/images/leaderboard.png" width="800"/>
</div>

Performance of existing MLLMs on MM-IFEval. We report the accuracy of easy and difficult problems and the average accuracy across all problems. The C-Level and P-Level refer to the compose-level and perception-level problems, respectively. The best performance in each section is highlighted in bold.

## 🚀 Evaluate on MM-IFEval
### Option 1 (Recommended): Evaluation using [VLMEvalkit](https://github.com/OpenCompass/VLMEvalkit)

```bash
# Note: Default snapshot of judge model (gpt-4o) in VLMEvalkit is currently gpt-4o-2024-05-13.

# When running with `python`, only one VLM instance is instantiated.
# API MODEL
python run.py --data MMIFEval --model GPT4o_MINI --reuse --verbose --api-nproc 8
# HF MODEL
python run.py --data MMIFEval --model Qwen2.5-VL-7B-Instruct --reuse --verbose --api-nproc 8

# When running with `torchrun`, one VLM instance is instantiated on each GPU. It can speed up the inference.
# HF MODEL
torchrun --nproc-per-node=2 run.py --data MMIFEval --model Qwen2.5-VL-7B-Instruct --reuse --verbose --api-nproc 8
```



### Option 2 : Evaluation using this repo

#### 1. Environment Setup

see requirements.txt

#### 2. Run Evaluation Script
```python
# Step1: finish the config below in eval_mmifeval/sh_scripts/multi_run_inf_and_score.sh
# <---- param settings ---->
PROJECT_DIR=
CONDA_ACTIVATE_PATH=
export HF_HOME=
model_bench_pairs=(
    "Qwen2-VL-7B-Instruct C-Level 8 qwen_vl HF"
    "Qwen2-VL-7B-Instruct P-Level 8 qwen_vl HF"
)
# <---- param settings ---->

# Step2: run the script
zsh eval_mmifeval/sh_scripts/multi_run_inf_and_score.sh
```


## 📝 Citation
```bibtex
@article{ding2025mm,
  title={MM-IFEngine: Towards Multimodal Instruction Following},
  author={Ding, Shengyuan and Wu, Shenxi and Zhao, Xiangyu and Zang, Yuhang and Duan, Haodong and Dong, Xiaoyi and Zhang, Pan and Cao, Yuhang and Lin, Dahua and Wang, Jiaqi},
  journal={arXiv preprint arXiv:2504.07957},
  year={2025}
}
```
