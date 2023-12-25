# 基于远端服务器的Llama2 CPU部署方案
## 下载申请

Llama2作为Meta发布的开源大语言模型，可免费用于学术研究或商业用途。本章主要叙述如何在本地（或自己的远程服务器）Linux系统上申请，部署以及运行Llama2模型的demo。

**申请Llama2许可**

要想使用Llama2，首先需要向meta公司申请使用许可，否则你将无法下载到Llama2的模型权重。

申请地址：[Llama2-7B-chat](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)

<img width="800" alt="image" src="https://github.com/Hennnnnry/Llama2_deployment/assets/96227995/420825f9-2d91-4ad2-9fa6-1aeb2a2ab81a">

填入对应信息（主要是邮箱）后，勾选页面最底部的 “I accept the terms and conditions”，点击 “Accept and Continue”，跳转到下图界面即可。

<img width="800" alt="image" src="https://github.com/Hennnnnry/Llama2_deployment/assets/96227995/5a2d46db-495f-4ce4-bb1e-3146aac0bcaa">

一段时间后，你将会收到一封邮件，框框部分以即为下面下载模型时需要验证的内容。

<img width="500" alt="image" src="https://github.com/Hennnnnry/Llama2_deployment/assets/96227995/b3bbc790-5b41-4b93-9cdb-0787bfbf8ecf">


申请后下载URL会发送到填写的邮箱，需要等几分钟。官方将在你申请后的24小时之内批准，要24小时内下载完成，否则权限可能随时被收回。

## 服务器 & 操作系统

**1. 安装 Conda 环境**

* 下载Anaconda安装包

进入自己的主目录中，下载Anaconda安装包。

```
wget -c https://mirrors.bfsu.edu.cn/anaconda/archive/Anaconda3-2022.10-Linux-x86_64.sh --no-check-certificate
```

你可以浏览镜像网站来查看其他版本的conda安装包，亦或者可以安装Miniconda（即Anaconda的纯命令行版本）。只需要用目标安装包的名称替换掉在上面的wget命令中Anaconda3-2022.10-Linux-x86_64.sh 即可。这里就以该安装包为例进行安装操作。

* 安装Anaconda

下载完成后使用ls指令应当可以在当前目录中看到自己下载的Anaconda安装包，名称通常是 Anaconda3-xxxx.xx-xxxxxx.sh 形式的文件。随后使用bash指令安装该安装包。

```
bash Anaconda3-2022.10-Linux-x86_64.sh
```

按下回车阅读协议，直到提示是否接受协议，输入yes，之后对安装路径没有要求的情况下无脑回车即可。

安装完成后会问是否要初始化，yes即可。

* 环境变量

重启服务器后，可以通过 conda -V 命令来检测conda是否被成功安装上。但此时系统大概率会提示找不到conda命令。我们需要激活一下已经修改的环境变量。

```
bash source ~/.bashrc
```

此时命令行最前端会显示(base)，再运行 conda -V ，此时应当可以看到系统输出conda的版本号。

至此我们便完成了conda环境的安装和配置。

**2. 创建虚拟环境**

在这一步中我们需要使用conda创建一个虚拟环境以运行Llama2模型。首先运行下列命令，即可创建一个名为llama， python版本为3.10的虚拟环境。这里建议安装的python版本为3.9及以上。

```
bash conda create -n llama python=3.10 
```

在``Proceed ([y]/n)? ``后输入 y ，即可自动创建环境。等待创建完成后，我们可以使用下列命令查看当前拥有的环境，并进入到llama环境中去。

```
# 查看当前存在的环境
conda env list
# 激活目标虚拟环境(llama)
conda activate llama
```

执行成功后，命令行前部的(base)标识会变成目标环境的名字，即(llama) 。

**3. 安装依赖**

克隆meta在github的llama项目：

```
sudo git clone https://github.com/facebookresearch/llama.git
```

里面有自动部署文件，clone完成后，进llama文件夹，执行：

```
cd llama
sudo pip install -e .
```

注意后面有一个“.”
安装过程中，torch相关文件较大（torchrun是PyTorch提供的用于分布式训练的命令行工具），建议使用国内源事先单独安装,这里使用清华源：

```
pip install --upgrade torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple fastNLP
```

安装完后就可以下模型了。

## 模型文件下载

**1. Meta版本**

检查一下邮箱的Meta AI发来的邮件，有基本的介绍和下载URL。

llama2有7B、13B和70B三个版本，分别为70 亿、130 亿和 700 亿三种参数变体，参数越多对配置要求越高。
每个版本有可调参的版本和chat版本，我这里选择第一个7B的版本。

执行llama文件夹里的下载脚本：

```
bash download.sh
```

按照提示，贴上邮件内提供的下载的URL，选择需要下载的版本，然后等待下载完成，7B的文件13G比较大得等半天，其他的版本更大。之后就可以开始体验了。

**2. Hugging Face版本**

如果你已经熟悉并习惯使用huggingface的模型以及训练方式。Llama2也有huggingface的版本可供使用。

需要注意的是，使用Meta官方的huggingface版本的Llama2模型也需要向Meta公司申请验证链接，此外还需使用申请账号登入Hugging Face官网，进入Meta Llama 2页面，同意用户协议后并递交申请，等待Meta公司的审核通过。

Ps：Meta申请链接中填写的邮箱要与Hugging Face的注册邮箱保持一致

递交申请后，审核需要一段时间，待申请通过后，你可以在模型页面内看到并下载所有的相关文件。

<img width="800" alt="image" src="https://github.com/Hennnnnry/Llama2_deployment/assets/96227995/c1154a9e-7cf4-4a46-98ba-abc652a1d0fb">

中文预训练模型位置（无需申请许可）：

<img width="800" alt="image" src="https://github.com/Hennnnnry/Llama2_deployment/assets/96227995/979e416e-a8d1-48c8-9f01-f55f8cc24d1d">



下载地址：[Llama2-Chinese-13B-16K](https://huggingface.co/hfl/chinese-alpaca-2-13b-16k-gguf/tree/main)


中文大模型参考网址：

https://github.com/ymcui/Chinese-LLaMA-Alpaca-2/?tab=readme-ov-file

https://github.com/FlagAlpha/Llama2-Chinese?tab=readme-ov-file

## CPU部署

大模型的研究分为训练和推理两个部分。训练的过程，实际上就是在寻找模型参数，使得模型的损失函数最小化，推理结果最优化的过程。训练完成之后，模型的参数就固定了，这时候就可以使用模型进行推理，对外提供服务。

llama.cpp 主要解决的是推理过程中的性能问题。主要有两点优化：

llama.cpp 使用的是 C 语言写的机器学习张量库 ggml
llama.cpp 提供了模型量化的工具
计算类 Python 库的优化手段之一就是使用 C 重新实现，这部分的性能提升非常明显。另外一个是量化，量化是通过牺牲模型参数的精度，来换取模型的推理速度。llama.cpp 提供了大模型量化的工具，可以将模型参数从 32 位浮点数转换为 16 位浮点数，甚至是 8、4 位整数。

除此之外，llama.cpp 还提供了服务化组件，可以直接对外提供模型的 API 。

由于核显性能不够，因此需要借助llama.cpp对模型进行量化处理，最后用cpu来运行模型，不需要显卡。

**1. Meta版本部署**

首先从github克隆llama.cpp到本地，这是一个文件夹，不是c++文件。

```
git clone https://github.com/ggerganov/llama.cpp
```

编译文件，llama.cpp文件夹中的makefile已经写好，直接进文件夹make就行：

```
cd llama.cpp 
make
```

编译后生成./main文件和./quantize二进制文件。

建立一个单独的文件夹来进行操作：

```
mkdir my-models/7B
```

然后进行文件转移，将llama/llama-2-7b文件夹中的consolidated.00.pth模型文件和配置文件params.json放到7B文件夹，llama/tokenizer.model放到my-models/

```
mv llama-2-7b/consolidated.00.pth llama.cpp/my-models/7B
mv llama-2-7b/params.json llama.cpp/my-models/7B
mv tokenizer.model llama.cpp/my-models/
```

将.pth文件转换为FP16格式/FP32格式，这一步要跑一会儿。

```
python3 convert.py my-models/7B/
```

对FP16/FP32模型进行4-bit量化，生成量化模型文件路径为my-models/7B/ggml-model-q4_0.bin

```
./quantize ./my-models/7B/ggml-model-f32.bin ./my-models/7B/ggml-model-q4_0.bin q4_0
```

然后就可以开始运行玩耍了，运行./main二进制文件：

```
./main -m my-models/7B/ggml-model-q4_0.bin --color -f prompts/alpaca.txt -ins -c 2048 --temp 0.2 -n 256 --repeat_penalty 1.1
```

**2. Hugging Face版本部署**

在 huggingface 上找到合适格式的模型，下载至 llama.cpp 的 models 目录下。

```
git clone https://huggingface.co/4bit/Llama-2-7b-chat-hf ./models/Llama-2-7b-chat-hf
```

llama.cpp 项目下带有 requirements.txt 文件，直接安装依赖即可。

```
pip install -r requirements.txt
```

* 转换模型

```
python convert.py ./models/Llama-2-7b-chat-hf --vocabtype spm
```

vocabtype 指定分词算法，默认值是 spm，如果是 bpe，需要显示指定。

* 量化模型

使用 quantize 量化模型，quantize 提供各种精度的量化。

```
./quantize

usage: ./quantize [--help] [--allow-requantize] [--leave-output-tensor] model-f32.gguf [model-quant.gguf] type [nthreads]

  --allow-requantize: Allows requantizing tensors that have already been quantized. Warning: This can severely reduce quality compared to quantizing from 16bit or 32bit
  --leave-output-tensor: Will leave output.weight un(re)quantized. Increases model size but may also increase quality, especially when requantizing

Allowed quantization types:
   2  or  Q4_0   :  3.56G, +0.2166 ppl @ LLaMA-v1-7B
   3  or  Q4_1   :  3.90G, +0.1585 ppl @ LLaMA-v1-7B
   8  or  Q5_0   :  4.33G, +0.0683 ppl @ LLaMA-v1-7B
   9  or  Q5_1   :  4.70G, +0.0349 ppl @ LLaMA-v1-7B
  10  or  Q2_K   :  2.63G, +0.6717 ppl @ LLaMA-v1-7B
  12  or  Q3_K   : alias for Q3_K_M
  11  or  Q3_K_S :  2.75G, +0.5551 ppl @ LLaMA-v1-7B
  12  or  Q3_K_M :  3.07G, +0.2496 ppl @ LLaMA-v1-7B
  13  or  Q3_K_L :  3.35G, +0.1764 ppl @ LLaMA-v1-7B
  15  or  Q4_K   : alias for Q4_K_M
  14  or  Q4_K_S :  3.59G, +0.0992 ppl @ LLaMA-v1-7B
  15  or  Q4_K_M :  3.80G, +0.0532 ppl @ LLaMA-v1-7B
  17  or  Q5_K   : alias for Q5_K_M
  16  or  Q5_K_S :  4.33G, +0.0400 ppl @ LLaMA-v1-7B
  17  or  Q5_K_M :  4.45G, +0.0122 ppl @ LLaMA-v1-7B
  18  or  Q6_K   :  5.15G, -0.0008 ppl @ LLaMA-v1-7B
   7  or  Q8_0   :  6.70G, +0.0004 ppl @ LLaMA-v1-7B
   1  or  F16    : 13.00G              @ 7B
   0  or  F32    : 26.00G              @ 7B
```

执行量化命令。

```
./quantize ./models/Llama-2-7b-chat-hf/ggml-model-f16.gguf ./models/Llama-2-7b-chat-hf/ggml-model-q4_0.gguf Q4_0
```

量化之后，模型的大小从 13G 降低到 3.6G，但模型精度从 16 位浮点数降低到 4 位整数。

* 模型推理

在 llama.cpp 项目的根目录，编译源码之后，执行下面的命令，使用模型进行推理。

```
./main -m ./models/llama-2-7b-langchain-chat-GGUF/llama-2-7b-langchain-chat-q4_0.gguf -p "What color is the sun?" -n 1024

 What color is the sun?
 nobody knows. It’s not a specific color, more a range of colors. Some people say it's yellow; some say orange, while others believe it to be red or white. Ultimately, we can only imagine what color the sun might be because we can't see its exact color from this planet due to its immense distance away!
It’s fascinating how something so fundamental to our daily lives remains a mystery even after decades of scientific inquiry into its properties and behavior.” [end of text]
```

当然，也可以用上面量化的模型进行推理。

```
./main -m  ./models/Llama-2-7b-chat-hf/ggml-model-q4_0.gguf -p "What color is the sun?" -n 1024

What color is the sun?
 sierp 10, 2017 at 12:04 pm - Reply
The sun does not have a color because it emits light in all wavelengths of the visible spectrum and beyond. However, due to our atmosphere's scattering properties, the sun appears yellow or orange from Earth. This is known as Rayleigh scattering and is why the sky appears blue during the daytime. [end of text]
```

四位量化模型，在没有 GPU 的情况下，基本能够实现实时推理。敲完命令，按回车，就能看到模型的回复。

main 命令有一系列参数可选，其中比较重要的参数有：

-ins 交互模式，可以连续对话，上下文会保留
-c 控制上下文的长度，值越大越能参考更长的对话历史（默认：512）
-n 控制回复生成的最大长度（默认：128）
–temp 温度系数，值越低回复的随机性越小

* 交互模式下模型模型推理

交互模式下，以对话的形式，有上下文的连续使用大模型。

```
./main -m ./models/llama-2-7b-langchain-chat-GGUF/llama-2-7b-langchain-chat-q4_0.gguf -ins

> 世界上最大的鱼是什么？
卡加内利亚鲨为世界最大的鱼，体长达60英尺（18）。牠们的头部相当于一只小车，身体非常丑，腹部有两个气孔，气孔之间还有一个大口径的鳃，用于进行捕食。牠们通常是从水中搴出来到陆地上抓到的小鱼，然后产生大量液体以解脱自己的身体。

> 现在还有这种鱼吗？
作者所提到的“卡加内利亚鲨”，应该是指的是“卡加内利亚鳄”。卡加内利亚鳄是一种大型淡水肉食性鱼类，分布于欧洲和非洲部分区域。这种鱼的体长最大可达60英尺（18），是世界上已知最大的鱼之一。

不过，现在这种鱼已经消失了，因为人类对戒备和保护水生生物的意识程度低下，以及环境污染等多方面原因。
```

## 提供模型 API 服务

**1. 使用 llama.cpp server 提供 API 服务**

前面编译之后，会在 llama.cpp 项目的根目录下生成一个 server 可执行文件，执行下面的命令，启动 API 服务。

```
./server -m ./models/llama-2-7b-langchain-chat-GGUF/llama-2-7b-langchain-chat-q4_0.gguf --host 0.0.0.0 --port 8080
```

这样就启动了一个 API 服务，可以使用 curl 命令进行测试。

```
curl --request POST \
    --url http://localhost:8080/completion \
    --header "Content-Type: application/json" \
    --data '{"prompt": "What color is the sun?","n_predict": 512}'
```

**2. 使用第三方工具包提供 API 服务**

在 llamm.cpp 项目的首页 https://github.com/ggerganov/llama.cpp 中有提到各种语言编写的第三方工具包，可以使用这些工具包提供 API 服务，包括 Python、Go、Node.js、Ruby、Rust、C#/.NET、Scala 3、Clojure、React Native、Java 等语言的实现。

以 Python 为例，使用 llama-cpp-python 提供 API 服务。

* 安装依赖

```
pip install llama-cpp-python -i https://mirrors.aliyun.com/pypi/simple/
```

如果需要针对特定的硬件进行优化，就配置 “CMAKE_ARGS” 参数，详情请参数 https://github.com/abetlen/llama-cpp-python

* 启动 API 服务

```
python -m llama_cpp.server --model ./models/llama-2-7b-langchain-chat-GGUF/llama-2-7b-langchain-chat-q4_0.gguf
```

在启动的过程中，可能因缺失一些依赖导致失败，根据提示安装即可。如果提示包版本冲突，则需要单独创建一个虚拟 Python 环境，然后安装依赖。

* 使用 curl 测试 API 服务

```
curl -X 'POST' \
  'http://localhost:8000/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "messages": [
    {
      "content": "You are a helpful assistant.",
      "role": "system"
    },
    {
      "content": "Write a poem for Chinese?",
      "role": "user"
    }
  ]
}'
```

* 使用 openai 调用 API 服务

```
# -*- coding: utf-8 -*-

import openai
openai.api_key = 'random'
openai.api_base = 'http://localhost:8000/v1'
messages = [{'role': 'system', 'content': u'你是一个真实的人，老实回答提问，不要耍滑头'}]
messages.append({'role': 'user', 'content': u'你昨晚去哪里了'})
response = openai.ChatCompletion.create(
    model='random',
    messages=messages,
)
print(response['choices'][0]['message']['content'])
```

```
我没有去任何地方。
```

这里的 api_key、model 可以随便填写，但是 api_base 必须指向真实服务地址 http://localhost:8000/v1。
























