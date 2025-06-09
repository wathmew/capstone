import os
import json
import torch
import gc, contextlib, ray
from argparse import ArgumentParser
from tqdm import tqdm
from vllm import SamplingParams
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment

from utils import *

def vllm_cleanup(llm):
    del llm.llm_engine.model_executor
    del llm
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()

class LLM():
    def __init__(self, config):
        self.config = config
        self.model, self.tokenizer = load_model(
            config.model,
            n_gpu=config.n_gpu,
            seed=config.seed,
            distributed_executor_backend="mp",     # Add this flag
            disable_custom_all_reduce=True         # Add this flag
        )
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
        )

        self.icl = "" if not hasattr(config, "icl") else config.icl
        self.prompt = config.prompt

    def contextual_translate(self, dataloader, multiple_context=False):
        def extract_translation(results):
            translations = []
            for result in results:
                translation = result.split("'")[0]
                while translation.endswith("."):
                    translation = translation[:-1]
                if translation.startswith("1)") or translation.startswith("1."):
                    translation = translation[2:].strip()

                translation = translation.replace("\ufffd", "")
                translation = translation.replace("\u200b", "")

                translations.append(translation)
            return translations

        translated_data = []
        for batch_idx, batch in tqdm(enumerate(dataloader)):
            prompt_batch = [
                self.prompt % (s, t)
                for s, t in zip(batch.batch_trans, batch.batch_text)
            ]
            prompt_batch = [self.icl + p for p in prompt_batch]

            results = self.model.generate(
                prompt_batch, self.sampling_params, use_tqdm=False
            )
            generations = [r.outputs[0].text for r in results]

            translated_batch = extract_translation(generations)
            translated_data.extend(translated_batch)

        return translated_data

def main():
    # configuration
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()
    config, xgear_config = load_config(args)

    # Set seed
    set_seed(config.seed)

    if config.task == "eae":
        # load data – adapt according to data format
        src_data = load_data(xgear_config)
        src_lang = config.src_lang

        # load 3rd-party translated data
        with open(
            f"{config.translated_text_folder}/translated_{config.split}.json", "r", encoding="utf-8"
        ) as f:
            translated_text = json.load(f)[0]

        processed_src, dataloader = process_contextual_data_contextual(
            src_data,
            translated_text,
            src_lang,
            xgear_config,
            config.max_batch_size,
            config.src_lang,
        )

        translated_data = None
        if config.load_orig_translations and os.path.exists(
            f"{config.translations_folder}/translated_{config.split}.json"
        ):
            with open(
                f"{config.translations_folder}/translated_{config.split}.json", "r", encoding="utf-8"
            ) as f:
                translated_data = json.load(f)
        else:
            model = LLM(config)

            try:

                # Independently translate text, trigger, and arguments
                translated_data = []
                translated_data.append(translated_text)
                translated_data.append(model.contextual_translate(dataloader))
                translated_data.append(processed_src[2])
                translated_data.append(processed_src[3])
            finally:
                vllm_cleanup(model.model)  # CLEANUP
            # Save original translations
            if config.save_orig_translations:
                os.makedirs(config.translations_folder, exist_ok=True)
                with open(
                    f"{config.translations_folder}/translated_{config.split}.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(translated_data, f, ensure_ascii=False)

        # Create target folder
        os.makedirs(config.tgt_folder, exist_ok=True)

        # Save data – adapt according to data format
        create_data_and_save_eae(processed_src, translated_data, config)

    elif config.task == "ner":
        # load data – adapt according to data format
        with open(
            f"{config.translated_text_folder}/source_{config.split}.json", "r", encoding="utf-8"
        ) as f:
            src_data = json.load(f)
        with open(
            f"{config.translated_text_folder}/translated_{config.split}.json", "r", encoding="utf-8"
        ) as f:
            old_translated_text = json.load(f)[0]
            translated_text = [clean_text(t) for t in old_translated_text]

        dataloader = process_data_contextual_ner(
            src_data, translated_text, config.max_batch_size
        )

        translated_data = None
        if config.load_orig_translations and os.path.exists(
            f"{config.translations_folder}/translated_{config.split}.json"
        ):
            with open(
                f"{config.translations_folder}/translated_{config.split}.json", "r", encoding="utf-8"
            ) as f:
                translated_data = json.load(f)
        else:
            model = LLM(config)
            try:

                # Independently translate text, entity
                translated_data = []
                translated_data.append(translated_text)
                translated_data.append(model.contextual_translate(dataloader))
                translated_data.append(src_data[2])
                translated_data.append(src_data[3])
            finally:
                vllm_cleanup(model.model)  # CLEANUP

            # Save original translations
            if config.save_orig_translations:
                os.makedirs(config.translations_folder, exist_ok=True)
                with open(
                    f"{config.translations_folder}/translated_{config.split}.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(translated_data, f, ensure_ascii=False)

        # Create target folder
        os.makedirs(config.tgt_folder, exist_ok=True)

        create_data_and_save_ner(
            translated_data,
            config,
            split=config.split,
            all_entity_included=config.labels_in_sent,
        )


if __name__ == "__main__":
    main()
