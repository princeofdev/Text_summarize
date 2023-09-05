"""
aggregate.py - module for aggregating text from multiple sources/multiple parts of a single source.
    Primary usage is through the BatchAggregator class.

How it works:
1. We tell the language model to do it.
2. The language model does it.
3. Yaay!
"""
import logging
import pprint as pp
import time

import torch
from transformers import GenerationConfig, pipeline

from utils import compare_model_size

# Setting up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class BatchAggregator:
    """
    BatchAggregator is a class for aggregating text from multiple sources.

    Usage:
    >>> from aggregate import BatchAggregator
    >>> aggregator = BatchAggregator()
    >>> agg = aggregator.infer_aggregate(["This is a test", "This is another test"])
    >>> print(agg)
    """

    GENERIC_CONFIG = GenerationConfig(
        num_beams=8,
        early_stopping=True,
        do_sample=False,
        min_new_tokens=32,
        max_new_tokens=256,
        repetition_penalty=1.1,
        length_penalty=1.4,
        no_repeat_ngram_size=4,
        encoder_no_repeat_ngram_size=5,
    )
    CONFIGURED_MODELS = [
        "pszemraj/bart-large-mnli-dolly_hhrlhf-v1",
        "pszemraj/bart-base-instruct-dolly_hhrlhf",
        "pszemraj/flan-t5-large-instruct-dolly_hhrlhf",
        "pszemraj/flan-t5-base-instruct-dolly_hhrlhf",
    ]  # these have generation configs defined for this task in their model repos

    DEFAULT_INSTRUCTION = "Write a comprehensive yet concise summary that pulls together the main points of the following text:"

    def __init__(
        self,
        model_name: str = "pszemraj/bart-large-mnli-dolly_hhrlhf-v1",
        force_cpu: bool = False,
        **kwargs,
    ):
        """
        __init__ initializes the BatchAggregator class.

        :param str model_name: model name to use, default: "pszemraj/bart-large-mnli-dolly_hhrlhf-v1"
        :param bool force_cpu: force the model to run on CPU, default: False
        """
        self.device = None
        self.is_compiled = False
        self.model_name = None
        self.aggregator = None
        self.force_cpu = force_cpu
        self.logger = logging.getLogger(__name__)
        self.init_model(model_name)

    def init_model(self, model_name: str) -> None:
        """
        Initialize the model.

        :param model_name: The name of the model to use.
        """
        # Free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info(f"Setting model to {model_name}")
        self.model_name = model_name
        self.aggregator = self._create_pipeline(model_name)
        self._configure_model()
        # update the generation config with the specific tokenizer
        tokenizer_params = {
            "decoder_start_token_id": 0
            if "t5" in model_name.lower()
            else self.aggregator.tokenizer.eos_token_id,
            "eos_token_id": 1
            if "t5" in model_name.lower()
            else self.aggregator.tokenizer.eos_token_id,
            "pad_token_id": 0
            if "t5" in model_name.lower()
            else self.aggregator.tokenizer.pad_token_id,
        }
        self.update_generation_config(**tokenizer_params)

    def _create_pipeline(
        self, model_name: str = "pszemraj/bart-large-mnli-dolly_hhrlhf-v1"
    ) -> pipeline:
        """
        _create_pipeline creates a pipeline for the model.

        :param str model_name: model name to use, default: "pszemraj/bart-large-mnli-dolly_hhrlhf-v1"
        :return pipeline: the pipeline for the model

        :raises Exception: if the pipeline cannot be created
        """
        self.device = 0 if torch.cuda.is_available() and not self.force_cpu else -1
        try:
            self.logger.info(
                f"Creating pipeline with model {model_name} on device {self.device}"
            )
            return pipeline(
                "text2text-generation",
                model_name,
                device=self.device,
                torch_dtype=torch.float32,
            )
        except Exception as e:
            self.logger.error(f"Failed to create pipeline: {e}")
            raise

    def _configure_model(self):
        """
        Configure the model for generation.
        """
        try:
            self.aggregator.model = torch.compile(self.aggregator.model)
            self.is_compiled = True
        except Exception as e:
            self.logger.warning(f"Could not compile model with Torch 2.0: {e}")

        if self.model_name not in self.CONFIGURED_MODELS:
            self.logger.info("Setting generation config to general defaults")
            self._set_default_generation_config()
        else:
            try:
                self.logger.info("Loading generation config from hub")
                self.aggregator.model.generation_config = (
                    GenerationConfig.from_pretrained(self.model_name)
                )
            except Exception as e:
                self.logger.warning(
                    f"Could not load generation config, using defaults: {e}"
                )
                self._set_default_generation_config()

        self.logger.info(self.aggregator.model.generation_config.to_json_string())

    def _set_default_generation_config(self):
        """
        Set the default generation configuration for the model.
        """
        self.aggregator.model.generation_config = self.GENERIC_CONFIG

        if (
            "large"
            or "xl" in self.model_name.lower()
            or compare_model_size(self.model_name, 500)
        ):
            upd = {"num_beams": 4}
            self.update_generation_config(**upd)

    def update_generation_config(self, **kwargs):
        """
        Update the generation configuration with the specified parameters.

        Args:
            **kwargs: The parameters to update in the generation configuration.
        """
        self.logger.info(f"Updating generation config with {pp.pformat(kwargs)}")

        self.aggregator.model.generation_config.update(**kwargs)

    def get_generation_config(self) -> dict:
        """
        Get the current generation configuration.

        Returns:
            dict: The current generation configuration.
        """
        return self.aggregator.model.generation_config.to_dict()

    def update_loglevel(self, level: str = "INFO"):
        """
        Update the log level.

        Args:
            level (str): The log level to set. Defaults to "INFO".
        """
        self.logger.setLevel(level)

    def infer_aggregate(
        self,
        text_list: list,
        instruction: str = DEFAULT_INSTRUCTION,
        **kwargs,
    ) -> str:
        f"""
        infer_aggregate - infers a consolidated summary from a list of texts.

        Args:
            text_list (list): The texts to summarize.
            instruction (str): The instruction for the summary. Defaults to {self.DEFAULT_INSTRUCTION}.
            **kwargs: Additional parameters to update in the generation configuration.

        Returns:
            The generated summary.
        """
        joined_text = "\n".join(text_list)
        prompt = f"{instruction}\n\n{joined_text}\n"
        if kwargs:
            self.update_generation_config(**kwargs)
        st = time.perf_counter()
        self.logger.info(f"inference on {len(text_list)} texts ...")
        result = self.aggregator(
            prompt,
            generation_config=self.aggregator.model.generation_config,
        )[0]["generated_text"]
        self.logger.info(f"Done. runtime:\t{round(time.perf_counter() - st, 2)}s")
        self.logger.info(
            f"Input tokens:\t{self.count_tokens(prompt)}. Output tokens:\t{self.count_tokens(result)}"
        )
        self.logger.debug(f"Generated text:\n{result}")

        return result

    def count_tokens(self, text: str) -> int:
        """count the number of tokens in a text"""
        return (
            len(self.aggregator.tokenizer.encode(text, truncation=False, padding=False))
            if text
            else 0
        )
