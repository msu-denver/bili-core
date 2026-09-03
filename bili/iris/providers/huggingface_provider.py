"""Local HuggingFace provider for bili-core IRIS.

Wraps ``langchain_huggingface.ChatHuggingFace`` behind the
:class:`LLMProvider` interface for loading models from the HuggingFace Hub or
a local directory.

Note: ``ChatHuggingFace`` does not currently support automatic tool calling.
See ``bili.iris.config.llm_config.LLM_MODELS`` for the ``supports_tools: False``
flag on this provider's entries.

Heavy dependencies
------------------
``torch``, ``transformers``, and ``langchain_huggingface`` are imported inside
:meth:`HuggingFaceProvider.load` to avoid loading heavy ML frameworks when
this provider is not in use.
"""

import gc
import logging
from typing import Any, Optional

from .base import LLMProvider

LOGGER = logging.getLogger(__name__)


# pylint: disable=too-few-public-methods
class HuggingFaceProvider(LLMProvider):
    """Local HuggingFace in-process model provider.

    Accepted kwargs
    ---------------
    model_name : str
        HuggingFace model name or local directory path.
    max_tokens : int, optional
        Maximum tokens to generate (``max_new_tokens``).
    temperature : float, optional
        Sampling temperature.
    top_p : float, optional
        Nucleus sampling probability.
    top_k : int, optional
        Top-k sampling limit.
    seed : int, optional
        Random seed for reproducibility.
    """

    def load(  # pylint: disable=arguments-differ,too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        model_name: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        **_extra: Any,
    ) -> Any:
        """Load and return a ``ChatHuggingFace`` instance.

        Loads the model into memory using ``AutoModelForCausalLM`` with
        ``device_map="auto"`` so the framework selects the best available
        device (CUDA, Apple MPS, or CPU).

        :param model_name: HuggingFace model ID or local path.
        :returns: A ``ChatHuggingFace`` instance wrapping a text-generation
            pipeline.
        :raises ImportError: If ``torch``, ``transformers``, or
            ``langchain_huggingface`` are not installed.
        """
        # pylint: disable=import-outside-toplevel
        import torch
        from langchain_huggingface.chat_models.huggingface import (
            ChatHuggingFace,
            HuggingFacePipeline,
        )
        from transformers import AutoModelForCausalLM, pipeline

        from bili.iris.loaders.tokenizer_loader import load_huggingface_tokenizer

        LOGGER.info("Loading HuggingFace model from %s", model_name)

        tokenizer = load_huggingface_tokenizer(model_name)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            offload_folder="/tmp/model_offload",
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        generation_config: dict = {
            "do_sample": True,
            "repetition_penalty": 1.176,
        }
        if top_p is not None:
            generation_config["top_p"] = top_p
        if top_k is not None:
            generation_config["top_k"] = top_k
        if seed is not None:
            generation_config["seed"] = seed
        if max_tokens is not None:
            generation_config["max_new_tokens"] = max_tokens
        if temperature is not None:
            generation_config["temperature"] = temperature

        text_pipeline = pipeline(
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            task="text-generation",
            return_full_text=False,
            model=model,
            tokenizer=tokenizer,
            **generation_config,
        )

        hf_pipeline = HuggingFacePipeline(pipeline=text_pipeline)
        chat_hf = ChatHuggingFace(llm=hf_pipeline)
        LOGGER.debug(chat_hf)
        return chat_hf
