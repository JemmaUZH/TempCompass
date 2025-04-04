<h2 align="center"> <a href="https://arxiv.org/abs/2403.00476">TempCompass: A benchmark to evaluate the temporal perception ability of Video LLMs</a></h2>

<div align="center">
    <a href='https://arxiv.org/abs/2403.00476'><img src='https://img.shields.io/badge/ArXiv-2403.00476-red'></a>
    <a href='https://llyx97.github.io/tempcompass/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
    <a href='https://huggingface.co/spaces/lyx97/TempCompass'><img src='https://img.shields.io/badge/🤗_Hugging_Face-Leaderboard-blue'></a>
    <a href='https://huggingface.co/datasets/lmms-lab/TempCompass'><img src='https://img.shields.io/badge/🤗_Hugging_Face-Datasets-green'></a>
</div>

<div>
<div align="center">
    <a href='https://llyx97.github.io/' target='_blank'>Yuanxin Liu<sup>1*</sup></a>&emsp;
    <a href='https://lscpku.github.io/' target='_blank'>Shicheng Li<sup>1*</sup></a>&emsp;
    <a href='https://liuyi-pku.github.io/' target='_blank'>Yi Liu<sup>1</sup></a>&emsp;
    Yuxiang Wang<sup>1</sup>&emsp;
    <a href='https://renshuhuai-andy.github.io/' target='_blank'>Shuhuai Ren<sup>1</sup></a>&emsp;
    </br>
    <a href='https://lilei-nlp.github.io/' target='_blank'>Lei Li<sup>2</sup></a>&emsp;
    <a href='https://pkucss.github.io/' target='_blank'>Sishuo Chen<sup>1</sup></a>&emsp;
    <a href='https://xusun26.github.io/' target='_blank'>Xu Sun<sup>1</sup></a>&emsp;
    <a href='https://houlu369.github.io/' target='_blank'>Lu Hou<sup>3</sup></a>
</div>
<div>
<div align="center">
    <sup>1</sup>Peking University&emsp;
    <sup>2</sup>The University of Hong Kong&emsp;
    <sup>3</sup>Huawei Noah’s Ark Lab
</div>
<div align="center">
    <sup>*</sup>Equal Contribution
</div>

## 📢 News

**[2024-10-30]** 🎉🎉🎉 TempCompass is integrated into [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).

**[2024-08-30]** Results of [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct), [GPT-4o](https://openai.com/index/hello-gpt-4o/), [MiniCPM-V-2.6](https://huggingface.co/openbmb/MiniCPM-V-2_6), [InternVL-2-8B](https://huggingface.co/OpenGVLab/InternVL2-8B), [LLaVA-OneVision-Qwen-2-7B](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov) and [InterLM-XComposer-2.5](https://huggingface.co/internlm/internlm-xcomposer2d5-7b) are added to the [leaderboard](https://huggingface.co/spaces/lyx97/TempCompass). GPT-4o establishes the new SoTA!

**[2024-08-08]** Results of [LLaVA-Next-Video](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/inference/docs/LLaVA-NeXT-Video_0716.md), [VILA-1.5](https://github.com/NVlabs/VILA) and [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA) are added to the [leaderboard](https://huggingface.co/spaces/lyx97/TempCompass).

**[2024-07]** 🎉🎉🎉 TempCompass is integrated into [LMMs-Eval](https://lmms-lab.github.io/posts/lmms-eval-0.2/). See [here](#lmms-eval) for usage examples.

**[2024-06-11]** Result of [Reka-core](https://www.reka.ai/news/reka-core-our-frontier-class-multimodal-language-model) is added to the [leaderboard](https://huggingface.co/spaces/lyx97/TempCompass).

**[2024-05-25]** [TempCompass Leaderboard](https://huggingface.co/spaces/lyx97/TempCompass) is available on HuggingFace Space 🤗.

**[2024-05-16]** 🎊🎊🎊 TempCompass is accepted at ACL 2024 Findings!

**[2024-04-14]** Evaluation [result](#eval_result) of [Gemini-1.5-pro](https://deepmind.google/technologies/gemini/pro/), the current SOTA Video LLM, is add.

**[2024-03-23]** The [answer prompt](#answer_prompt) is improved to better guide Video LLMs to follow the desired answer formats. The [evaluation code](#eval) now provides an option to disable the use of ChatGPT.

**[2024-03-12]** 🔥🔥🔥 The evaluation code is released now! Feel free to evaluate your own Video LLMs.

## 🏆 LeaderBoard
![](./assets/leaderboard.png)

## ✨ Highlights
### Diverse Temporal Aspects and Task Formats
- TempCompass encompasses a diverse set of temporal aspects (left) and task formats (right) to comprehensively evaluate the temporal perception capability of Video LLMs.
![](./assets/overview.png)
### Conflicting Videos
- We construct conflicting videos to prevent the models from taking advantage of single-frame bias and language priors.
![](./assets/conflicting_videos.jpg)
  
- 🤔 Can your Video LLM correctly answer the following question for both two videos?
  
    <img src="./assets/1021488277.gif" alt="Raw Video" style="float: left; width: 49%; margin-right: 10px;">
    <img src="./assets/1021488277_reverse.gif" alt="Conflicting Video" style="float: left; width: 49%;">
    
    > What is happening in the video?    
    > A. A person drops down the pineapple    
    > B. A person pushes forward the pineapple    
    > C. A person rotates the pineapple    
    > D. A person picks up the pineapple

## 🚀 Quick Start
To begin with, clone this repository and install some packages:
```shell
git clone https://github.com/llyx97/TempCompass.git
cd TempCompass
pip install -r requirements.txt
```

### Data Preparation
**1. Task Instructions**

The task instructions can be found in `questions/`.

<details>
<summary><span id="instruct_gen"> Task Instruction Generation Procedure </span></summary>
    
1. Generate **Multi-Choice QA** instructions (`question_gen.py`). 

2. Manually validate quality and rectify.

3. Generate task instructions for **Yes/No QA** (`question_gen_yes_no.py`), **Caption Matching** (`question_gen_caption_match.py`) and **Caption Generation** (`question_gen_captioning.py`), based on manually rectified **Multi-Choice QA** instructions.
   
4. Manually validate quality and rectify.
</details>

**2. Videos**

All the processed videos can be downloaded from [google drive](https://drive.google.com/file/d/1b0ZIeRqhrUpQYxoCN_Ym_e0UW05cckYJ/view?usp=sharing) or [huggingface](https://huggingface.co/datasets/lmms-lab/TempCompass).

<details>
<summary><span id="instruct_gen"> As an alternative, you can also download the raw videos and process them yourself </span></summary>

Run the following commands. The videos will be saved to `videos/`.
```shell
cd utils
python download_video.py    # Download raw videos
python process_videos.py    # Construct conflicting videos
```

**Note:** If you encounter a `MoviePy error` when running the processing script, please refer to this [issue](https://github.com/llyx97/TempCompass/issues/4).
</details>

### Run Inference
We use [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA) and [Gemini](https://github.com/google-gemini/cookbook/blob/98a74b3cde77e518032928acec2fab8b8f3b41be/preview/file-api/File_API_Video.ipynb) as examples to illustrate how to conduct MLLM inference on our benchmark.

**1. Video-LLaVA**

Enter `run_video_llava` and install the environment as instructed.

Then run the following commands. The prediction results will be saved to `predictions/video-llava/<task_type>`.
```shell
# select <task_type> from multi-choice, yes_no, caption_matching, captioning
python inference_dataset.py --task_type <task_type>
```

**2. Gemini**

The inference script for gemini-1.5-pro is `run_gemini.ipynb`. It is recommended to run the script in [Google Colab](https://colab.research.google.com/).

### <span id="eval"> Run Evaluation </span>
After obtaining the MLLM predictions, run the following commands to conduct automatic evaluation. Remember to set your own `$OPENAI_API_KEY` in `utils/eval_utils.py`.

- **Multi-Choice QA**
`python eval_multi_choice.py --video_llm video-llava`

- **Yes/No QA**
`python eval_yes_no.py --video_llm video-llava`

- **Caption Matching**
`python eval_caption_matching.py --video_llm video-llava`

- **Caption Generation**
`python eval_captioning.py --video_llm video-llava`

**Tip**👉: Except for *Caption Generation*, you can set `--disable_llm` when running the scripts, which will disable chatgpt-based evaluation (i.e., entirely rely on rule-based evaluation). **This is useful when you do not want to use ChatGPT API and your MLLM is good at following the instruction to generate answers of specific format.**

The results of each data point will be saved to `auto_eval_results/video-llava/<task_type>.json` and the overall results on each temporal aspect will be printed out as follows:
```
{'action': 76.0, 'direction': 35.2, 'speed': 35.6, 'order': 37.7, 'attribute_change': 41.0, 'avg': 45.6}
{'fine-grained action': 58.8, 'coarse-grained action': 90.3, 'object motion': 36.2, 'camera motion': 32.6, 'absolute speed': 47.6, 'relative speed': 28.0, 'order': 37.7, 'color & light change': 43.6, 'size & shape change': 39.4, 'combined change': 41.7, 'other change': 38.9}
Match Success Rate=100.0
```

## <span id="lmms-eval"> LMMs-Eval Evaluation </span>
Here we provide an example of how to evaluate LLaVA-Next-Video on TempCompass, using lmms-eval.

**1. Clone the repo from [LLaVA-Next](https://github.com/LLaVA-VL/LLaVA-NeXT) and setup environments**
```
git clone https://github.com/LLaVA-VL/LLaVA-NeXT
cd LLaVA-NeXT
pip install -e .
```
**2. Run inference and evaluation in a single command**
```
accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model llavavid \
    --model_args pretrained=lmms-lab/LLaVA-NeXT-Video-32B-Qwen,conv_template=qwen_1_5,video_decode_backend=decord,max_frames_num=32,mm_spatial_pool_mode=average,mm_newline_position=grid,mm_resampler_location=after \
    --tasks tempcompass \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_vid_32B \
    --output_path ./logs/
```
You can also evaluate the performance on each task (e.g., multi-choice) seperately:
```
accelerate launch --num_processes 8 --main_process_port 12345 -m lmms_eval \
    --model llavavid \
    --model_args pretrained=lmms-lab/LLaVA-NeXT-Video-32B-Qwen,conv_template=qwen_1_5,video_decode_backend=decord,max_frames_num=32,mm_spatial_pool_mode=average,mm_newline_position=grid,mm_resampler_location=after \
    --tasks tempcompass_multi_choice \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_vid_32B \
    --output_path ./logs/
```
**3. Submit results to [TempCompass LeaderBoard](https://huggingface.co/spaces/lyx97/TempCompass)**

Place the lmms-eval outputs (`tempcompass_multi_choice.json`, `tempcompass_yes_no.json`, `tempcompass_caption_matching.json` and `tempcompass_captioning.json`) into the same folder and run this [script](https://huggingface.co/spaces/lyx97/TempCompass/blob/main/merge_eval_result.py):
```
python merge_eval_result.py
```
Then submit the output file `merged_result.json` to the leaderboard.

**Note:**
Currently, the evaluation results calculated by lmms-eval on specific temporal aspects might be incorrect (the average accuracy on each task is correct). To obtain the correct results, you can use this script: [acc_lmms_eval.py](https://github.com/llyx97/TempCompass/blob/main/utils/acc_lmms_eval.py) or submit the result to our leaderboard.

## 📈 Data Statistics
![](./assets/data_statistics.png)

## 📊 <span id="eval_result"> Evaluation Results </span>
The following figures present results of five representative Video LLMs. Results of more Video LLMs and Image LLMs can be found in our [paper](https://arxiv.org/abs/2403.00476) and the [leaderboard](https://huggingface.co/spaces/lyx97/TempCompass).

<img src="./assets/multi-choice.jpg" alt="Multi-Choice" style="float: left; width: 49%; margin-right: 10px;">
<img src="./assets/yes_no.jpg" alt="Yes/No" style="float: left; width: 49%;">
<img src="./assets/caption_matching.jpg" alt="Caption Matching" style="float: left; width: 49%; margin-right: 10px;">
<img src="./assets/captioning.jpg" alt="Caption Generation" style="float: left; width: 49%;">

### <span id="answer_prompt"> Answer Prompt </span>
We update the answer prompt for *Multi-Choice QA* and *Caption Matching*, from "Best Option:" to "Please directly give the best option:", which can better encourage MLLMs to directly select an option. As such, we can reduce the reliance on ChatGPT API, if an MLLM is good at following the instruction.

The success rate of rule-based matching is as follows:

**Multi-Choice QA**
|  | V-LLaVA | SPHINX-v2    | LLaMA-VID | Qwen-VL-Chat | PandaGPT  | Valley  |
| --- | --- | --- | --- | --- | --- | --- |
| old prompt | 37.9 | 99.6 | 62.9 | 46.8 | 6.4 | 3.5 |
| new prompt | 100 | 100 | 97.0 | 98.5 | 3.9 | 0.4 |

**Caption Matching**
|  | V-LLaVA | SPHINX-v2    | LLaMA-VID | Qwen-VL-Chat | PandaGPT  | Valley  |
| --- | --- | --- | --- | --- | --- | --- |
| old prompt | 76.6 | 89.3 | 44.5 | 91.6 | 30.7 | 11.2 |
| new prompt | 99.5 | 99.5 | 68.3 | 96.0 | 22.5 | 3.7 |

## TODOs
- [x] Upload scripts to collect and process videos.
- [x] Upload the code for automatic evaluation.
- [x] Upload the code for task instruction generation.

## License
This dataset is intended for academic research only. It is under [CC BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).

## Citation
```bibtex
@article{liu2024tempcompass,
  title   = {TempCompass: Do Video LLMs Really Understand Videos?},
  author  = {Yuanxin Liu and Shicheng Li and Yi Liu and Yuxiang Wang and Shuhuai Ren and Lei Li and Sishuo Chen and Xu Sun and Lu Hou},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2403.00476}
}
```
