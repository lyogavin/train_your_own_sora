## Latte Text to Video Training


Latte is by far the closest to SORA among the open-source video generation models.

Original Latte didn't provide text to video training code. **We reproduced the paper and implemented the text to video training based on the paper.**

Please find out more details from the paper:

> [**Latte: Latent Diffusion Transformer for Video Generation**](https://maxin-cn.github.io/latte_project/)<br>



 ![The architecture of Latte](visuals/architecture.svg)

## Improments

The following improvements are implemented to the training code:

* added the support of gradient accumulation (config: `gradient_accumulation_steps`)
* added valiation samples generation to generate (config: `validation`) testing videos in the training process
* added wandb support 
* added classifier-free guidance training (config: `cfg_random_null_text_ratio`)



## Step 1: setup the environment

First, download and set up the repo:

```bash
git clone https://github.com/lyogavin/Latte_t2v_training.git
conda env create -f environment.yml
conda activate latte
```

If you find it too complicated to setup the environment and solve all the package versions, cuda drivers, etc, you can try our vast.ai template [here](https://cloud.vast.ai/?ref_id=116659&template_id=38afe097c741a1e084afc68c473cde94).



## Step 2: download pretrained model

You can download the pretrained model as follows:

```bash
sudo apt-get install git-lfs # or: sudo yum install git-lfs
git lfs install

git clone --depth=1 --no-single-branch  https://huggingface.co/maxin-cn/Latte /root/pretrained_Latte/

```

## Step 4: prepare training data

Put video files in a directory and create a csv file to specify the prompt for each video.

The csv file format:


|video\_file\_name|prompt|
| ----------- | ----------- |
|VIDEO\_FILE\_001.mp4|PROMPT\_001|
|VIDEO\_FILE\_002.mp4|PROMPT\_002|
|...|...|



## Step 5: config

Config is in `configs/t2v/t2v_img_train.yaml` and it's pretty self-explanotary. 

A few config entries to note:

* point `video_folder` and `csv_path` to the path of training data
* point `pretrained_model_path` to the `t2v_required_models` directory of downloaded model.
* point `pretrained` to the t2v.pt file in the downloaded model
* You can change `text_prompt` under `validation` section to the testing validation prompts. During the training process every `ckpt_every` steps, it'll test generating videos based on the prompts and publish to wandb for you to checkout.


## Step 6: train!

```bash
./run_img_t2v_train.sh
```

## Cloud GPUs

We recommend vast.ai GPUs for training. 

We find it pretty good, low price, good network speed, wide range of GPUs to choose. Everything professionally optimized for AI training.

Feel free to use our [template](https://cloud.vast.ai/?ref_id=116659&template_id=38afe097c741a1e084afc68c473cde94) here where the environment is all ready to use.

## Inference

Reference original [repo](https://maxin-cn.github.io/latte_project/) for how to infer. 


## Stay Connected with Us

### Wechat public account


![group](https://github.com/lyogavin/Anima/blob/main/assets/wechat_pub_account.jpg?raw=true)


### Wechat group


<img src="https://github.com/lyogavin/Anima/blob/main/assets/wechat_group.png?raw=true" alt="group" style="width:260px;"/>

### Discord

[![Discord](https://img.shields.io/discord/1175437549783760896?logo=discord&color=7289da
)](https://discord.gg/2xffU5sn)

### Tech Blog

[![Website](https://img.shields.io/website?up_message=blog&url=https%3A%2F%2Fmedium.com%2F%40lyo.gavin&logo=medium&color=black)](https://medium.com/@lyo.gavin)

### Little RedBook

 ![redbook](visuals/redbook.png)


## Contribution 

Buy me a coffee please!  🙏

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://bmc.link/lyogavinQ)



## By: Anima AI


<img src="https://static.aicompose.cn/static/logo/animaai_logo.png?t=1696952962" alt="aiwrite" style="width:170px;"/>






