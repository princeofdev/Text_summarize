"""
summarize - a module for summarizing text using a model from the Hugging Face model hub
"""
import logging
import pprint as pp

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from utils import validate_pytorch2


def load_model_and_tokenizer(model_name: str) -> tuple:
    """
    load_model_and_tokenizer - load a model and tokenizer from a model name/ID on the hub

    :param str model_name: the model name/ID on the hub
    :return tuple: a tuple containing the model and tokenizer
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
    ).to(device)
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logging.info(f"Loaded model {model_name} to {device}")

    if validate_pytorch2():
        try:
            logging.info("Compiling model with Torch 2.0")
            model = torch.compile(model)
        except Exception as e:
            logging.warning(f"Could not compile model with Torch 2.0: {e}")
    else:
        logging.info("Torch 2.0 not detected, skipping compilation")

    return model, tokenizer


def summarize_and_score(
    ids, mask, model, tokenizer, is_general_attention_model=True, **kwargs
) -> tuple:
    """
    summarize_and_score - given a batch of ids and a mask, return a summary and a score for the summary

    Args:
        ids (): the batch of ids
        mask (): the attention mask for the batch
        model   (): the model to use for summarization
        tokenizer (): the tokenizer to use for summarization
        is_general_attention_model (bool, optional): whether the model is a general attention model. Defaults to True.
        **kwargs: any additional arguments to pass to the model
    Returns:
        tuple (str, float): the summary,  the score for the summary
    """

    ids = ids[None, :]
    mask = mask[None, :]

    input_ids = ids.to("cuda") if torch.cuda.is_available() else ids
    attention_mask = mask.to("cuda") if torch.cuda.is_available() else mask

    global_attention_mask = torch.zeros_like(attention_mask)
    # put global attention on <s> token
    global_attention_mask[:, 0] = 1

    if is_general_attention_model:
        summary_pred_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            output_scores=True,
            return_dict_in_generate=True,
            **kwargs,
        )
    else:
        summary_pred_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            output_scores=True,
            return_dict_in_generate=True,
            **kwargs,
        )
    summary = tokenizer.batch_decode(
        summary_pred_ids.sequences,
        skip_special_tokens=True,
        remove_invalid_values=True,
    )
    score = round(summary_pred_ids.sequences_scores.cpu().numpy()[0], 4)

    return summary, score


def summarize_via_tokenbatches(
    input_text: str,
    model,
    tokenizer,
    batch_length=2048,
    batch_stride=16,
    min_batch_length=512,
    **kwargs,
) -> list:
    """
    summarize_via_tokenbatches - summarize a long string via batches of tokens

    Args:
        input_text (str): the text to summarize
        model (): the model to use for summarization
        tokenizer (): the tokenizer to use for summarization
        batch_length (int, optional): the length of each batch. Defaults to 2048.
        batch_stride (int, optional): the stride of each batch. Defaults to 16. The stride is the number of tokens that overlap between batches.
        min_batch_length (int, optional): the minimum length of each batch. Defaults to 512.

        **kwargs: any additional arguments to pass to the model for inference
    Returns:
        list: a list of dictionaries containing the input tokens, the summary, and the summary score
    """

    logger = logging.getLogger(__name__)
    # log all input parameters
    if batch_length < min_batch_length:
        logger.warning(
            f"batch_length must be at least {min_batch_length}. Setting batch_length to {min_batch_length}"
        )
        batch_length = min_batch_length

    logger.info(f"input parameters:\n{pp.pformat(kwargs)}")
    logger.info(f"batch_length: {batch_length}, batch_stride: {batch_stride}")

    encoded_input = tokenizer(
        input_text,
        padding="max_length",
        truncation=True,
        max_length=batch_length,
        stride=batch_stride,
        return_overflowing_tokens=True,
        add_special_tokens=False,
        return_tensors="pt",
    )

    in_id_arr, att_arr = encoded_input.input_ids, encoded_input.attention_mask
    gen_summaries = []

    pbar = tqdm(total=len(in_id_arr))

    for _id, _mask in zip(in_id_arr, att_arr):
        result, score = summarize_and_score(
            ids=_id,
            mask=_mask,
            model=model,
            tokenizer=tokenizer,
            **kwargs,
        )
        score = round(float(score), 4)
        _sum = {
            "input_tokens": _id,
            "summary": result,
            "summary_score": score,
        }
        gen_summaries.append(_sum)
        logger.debug(f"Score for batch: {score}. num chars: {len(repr(result))}")
        logger.debug(f"Summary:\n\t{result}")
        pbar.update()

    pbar.close()

    return gen_summaries
