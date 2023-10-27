maxaillm
=========

finetune
########

MaxFineTuningEngine
*********
Interface to tune LLM models or transformers. This class allow to tune supported transformers weights by offered fine-tuning methods.

Args:
    - ``model_architecture (str)`` - name of a pretrained model
    - ``use_case (str)`` - type of use case. available options are 'text-generation', 'text-to-text'
    - ``provider (str)`` - provider for transformer and tokenizer weights. available options are 'hugging_face'
    - ``finetune_method (str)`` - fine-tune method to train transformer. available options are 'peft_lora'

>>> # create instance
>>> model_load_config = {
...     'model_name':'facebook/opt-125m'
... }
>>> tuner = MaxFineTuningEngine(
...     provider="hugging_face",
...     finetune_method="peft_lora",
...     use_case="text-generation",
...     model_architecture=model_load_config
... )

Methods
@@@@@@@

fine_tune
$$$$$$$$$
call the fine-tuning process based on the selected use case

Args:
    - ``tune_config (dict)`` - configuration to tune model
    - ``tune_data (dict)`` - dictionary of 'train' and 'test' datasets

>>> # start tuning
>>> tune_datasets = {
...     "train": None,
...     "test": None,
... }
>>> fine_tune_config = {
... "peft_lora_config": {
...         'r': 4,
...         'lora_alpha': 16,
...         'lora_dropout': 0.1,
...         'bias': 'none',
...         'task_type': 'CAUSAL_LM'
...     },
...    "trainer_config": {
...         'per_device_train_batch_size': 4,
...         'gradient_accumulation_steps': 32,
...         'warmup_steps': 20,
...         'num_train_epochs': 2,
...         'learning_rate': 2e-5,
...         'output_dir': 'sample-model-tune',
...         'max_steps': 5
...     }
... }
>>> tuner.fine_tune(tune_config=fine_tune_config, tune_data=tune_datasets)

save_model
$$$$$$$$$$
save the fine-tuned model to a specified path with provider export method implemented.

>>> tuner.save_model(
...     save_path="path/to/save",
...     save_config={}, 
...     token_size=20
... )

----------

finetune
########

LoraPeft
^^^^^^^^
tune base model with PEFT approach, using lora configs

Args:
    - ``model``
    - ``tokenizer``
    - ``use_case (str)``

Methods
@@@@@@@

finetune_model
$$$$$$$$$$$$$$
tune existing model with additional weights according to passed config

----------

Providers
*********

HuggingFace
^^^^^^^^^^^^
class to load model from MLflow or provider and save it back.

Args:
    - ``model_config (dict)`` - arguments required to load model, 'model_name' mandatory argument in dictionary
    - ``use_case (str)`` - type of use case. available options are 'text-generation', 'text-to-text'

Methods
@@@@@@@

load_pretrained_model
$$$$$$$$$$$$$$$$$$$$$$
load pretrained model from ml flow or provider

Args:
    - ``None``

Returns:
    - ``model``
    - ``tokenizer``

save_model
$$$$$$$$$$
save model into location at disk and ml flow if required

Args:
    - ``model``
    - ``tokenizer``
    - ``save_path (str)``
    - ``save_config (dict)``
    - ``token_size (int)`` - Defaults to ``None``.

Returns:
    - ``bool``
    


