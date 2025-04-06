
# Code for prompting LLMs for grammatical error correction

# Installation

I've tested the code with python=3.11.

1. Create a virtual environment (e.g. conda) and activate it

    ```
    conda create -n gec-prompting python=3.11 conda activate gec-prompting
    ```

1. Install pytorch==2.0.1

    `pip install torch==2.0.1 --index-url <https://download.pytorch.org/whl/cu118>`

1. Install the requirements using pip

    `pip install -r requirements.txt`

1. Download the spacy model for ERRANT.

    `python3 -m spacy download en_core_web_sm`

# Setup

Login to huggingface:

> huggingface-cli login

Setup your environment variables:

- create a file called .env at the top of your repo

    `touch path/to/gec-prompting/.env`

- copy and fill in/replace the following variables within the quotes and remove the #comments

    `OPENAI_API_KEY="" # Optional.`
    
    `COHERE_API_PROD_KEY="" # Optional. You can use a trial key here.`

    `HUGGINGFACE_HUB_CACHE="path/to/cache"`
    
    `HF_READ_TOKEN=""`  # See [HF access tokens](https://huggingface.co/docs/hub/en/security-tokens)
    
    `CORPORA="path/to/corpora"` # You could replace this with your own hardcoded path in each src/learner_datasets/*_dataset
    
    `TGI_VOLUME="path/to/$(whoami)/docker/data"`

# Data

You can download a zip with the data from [OSF GEC Prompting](https://osf.io/4b5tg/)

# Prompts

Prompts are hardcoded in src/prompts, with a different file for models that require special prompting syntax.

The order of the prompts across files **should** be the same. I.e. the first zero-shot prompt for OpenAI is the same as the first zero-shot prompt for Llama-2, but with different syntax.

# Prompting a LLM

The entry point is src/runhydra.py.

As there are a lot of settings (models x prompts x datasets), I've set this up to use [hydra configs](https://hydra.cc/docs/intro/) so that experiments are composable at the command line.

The configs are specified in gec-prompting/conf.

Running the script without any command line modifiers will use the default parameters in gec-prompting/conf/config.yaml:

`python src/runhydra.py`

The following sections demonstrate how to update the config settings at the command line.

## Specify a model

There are four *types* of model: OpenAI, Cohere, HuggingFace pipelines, HuggingFace TGI.

Each one has it's own set of parameters (see conf/model/params).

HuggingFace pipelines and TGI use the same model names, but their implementations use slightly different parameters.

HF models require the full model code. E.g. "bigscience/bloom-7b1" instead of "bloom-7b1"

For now, I've constrained the set of models and model names for additional validation. The full set of models are in src/models/__init\__.py

- I'm considering removing this constraint.

### Examples for each model type

`python src/runhydra.py model=hf model.name=bigscience/bloom-7b1`

`python src/runhydra.py model=tgi model.name=bigscience/bloom-7b1`

`python src/runhydra.py model=openai model.name=gpt-3.5-turbo-0613`

> **Note:** gpt-3.5-turbo-0613 is the default so you don't need to set the model name, and can instead just set `model=openai`

`python src/runhydra.py model=cohere model.name=command`

## Specifying batch sizes

There is a "batch" parameter but it really only works for OpenAI, Cohere, and HF TGI because they use asynchronous calls.

For these models, you can add the batch parameter with a value >1:

`python src/runhydra.py batch=100`

## Specify a prompt

Each prompt file (e.g. src/prompts/default_prompts.json) contains a key indicating the prompt type, followed by a list of prompt templates.

### to use zero-shot prompt #5

`python src/runhydra.py prompt_type=zero-shot prompt_index=5`

### to use 3-shot prompt #1

`python src/runhydra.py prompt_type=3-shot prompt_index=1`

## Specify a dataset

### for the W&I dev set

`python src/runhydra.py dataset=wibea dataset.split=dev`

### for the FCE dev set

`python src/runhydra.py dataset=fce dataset.split=dev`

## Composed examples

### Prompt gpt-3.5-turbo-0613 with a 3-shot prompt on W&I-dev, with batch 50

`python src/runhydra.py model=openai prompt_type=3-shot prompt_index=1 dataset=wibea dataset.split=dev batch=50`

### Prompt flan-t5-xxl using TGI, change the output directory, and batch 200

`python src/runhydra.py model=tgi model.name=google/flan-t5-xxl base_dir=/path/to/some/folder/my_exp batch=200`

# Evaluation

There is an evaluation script that:

- post-processes the hyp.txt model output to create a hyp_post.txt file
- runs ERRANT to create a hyp_post.m2 file
- uses ERRANT to compare the .m2 file to the reference
- if "conll" is in the model path, it runs the m2scorer
- if "jfleg" is in the model path, it runs the GLEU evaluation

For example:

`python src/evaluate/evaluate.py --base_dir "." --folder "model_output"`

By default, this script will evaluate all model outputs under the base_dir/folder.

The script has options to specify a subset of model outputs. For example, if you only want to evaluate W&I outputs:

`python src/evaluate/evaluate.py --base_dir "." --folder "model_output" --exps_pattern "*wibea\*"`
