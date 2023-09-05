"""
app.py - the main module for the gradio app for summarization

Usage:
    app.py [-h] [--share] [-m MODEL] [-nb ADD_BEAM_OPTION] [-batch TOKEN_BATCH_OPTION]
              [-level {DEBUG,INFO,WARNING,ERROR}]
Details:
    python app.py --help

Environment Variables:
    USE_TORCH (str): whether to use torch (1) or not (0)
    TOKENIZERS_PARALLELISM (str): whether to use parallelism (true) or not (false)
Optional Environment Variables:
    APP_MAX_WORDS (int): the maximum number of words to use for summarization
    APP_OCR_MAX_PAGES (int): the maximum number of pages to use for OCR
"""
import argparse
import contextlib
import gc
import logging
import os
import pprint as pp
import random
import re
import sys
import time
from pathlib import Path

os.environ["USE_TORCH"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%b-%d %H:%M:%S",
)

import gradio as gr
import nltk
import torch
from cleantext import clean
from doctr.models import ocr_predictor

from aggregate import BatchAggregator
from pdf2text import convert_PDF_to_Text
from summarize import load_model_and_tokenizer, summarize_via_tokenbatches
from utils import (
    contraction_aware_tokenize,
    extract_batches,
    load_example_filenames,
    remove_stagnant_files,
    remove_stopwords,
    saves_summary,
    textlist2html,
    truncate_word_count,
)

_here = Path(__file__).parent

nltk.download("punkt", force=True, quiet=True)
nltk.download("popular", force=True, quiet=True)

# Constants & Globals
MODEL_OPTIONS = [
    "pszemraj/long-t5-tglobal-base-16384-book-summary",
    "pszemraj/long-t5-tglobal-base-sci-simplify",
    "pszemraj/long-t5-tglobal-base-sci-simplify-elife",
    "pszemraj/long-t5-tglobal-base-16384-booksci-summary-v1",
    "pszemraj/pegasus-x-large-book-summary",
]  # models users can choose from
BEAM_OPTIONS = [2, 3, 4]  # beam sizes users can choose from
TOKEN_BATCH_OPTIONS = [
    1024,
    1536,
    2048,
    2560,
    3072,
]  # token batch sizes users can choose from

SUMMARY_PLACEHOLDER = "<p><em>Output will appear below:</em></p>"
AGGREGATE_MODEL = "MBZUAI/LaMini-Flan-T5-783M"  # model to use for aggregation

# if duplicating space: uncomment this line to adjust the max words
# os.environ["APP_MAX_WORDS"] = str(2048)  # set the max words to 2048
# os.environ["APP_OCR_MAX_PAGES"] = str(40)  # set the max pages to 40
# os.environ["APP_AGG_FORCE_CPU"] = str(1)  # force cpu for aggregation

aggregator = BatchAggregator(
    AGGREGATE_MODEL, force_cpu=os.environ.get("APP_AGG_FORCE_CPU", False)
)


def aggregate_text(
    summary_text: str,
    text_file: gr.inputs.File = None,
) -> str:
    """
    Aggregate the text from the batches.

        NOTE: you should probably include the BatchAggregator object as a fn arg if using this code

    :param batches_html: The batches to aggregate, in html format
    :param text_file: The text file to append the aggregate summary to
    :return: The aggregate summary in html format
    """
    if summary_text is None or summary_text == SUMMARY_PLACEHOLDER:
        logging.error("No text provided. Make sure a summary has been generated first.")
        return "Error: No text provided. Make sure a summary has been generated first."

    try:
        extracted_batches = extract_batches(summary_text)
    except Exception as e:
        logging.info(summary_text)
        logging.info(f"the batches html is: {type(summary_text)}")
        return f"Error: unable to extract batches - check input: {e}"
    if not extracted_batches:
        logging.error("unable to extract batches - check input")
        return "Error: unable to extract batches - check input"

    out_path = None
    if text_file is not None:
        out_path = text_file.name  # assuming name attribute stores the file path

    content_batches = [batch["content"] for batch in extracted_batches]
    full_summary = aggregator.infer_aggregate(content_batches)

    # if a path that exists is provided, append the summary with markdown formatting
    if out_path:
        out_path = Path(out_path)

        try:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write("\n\n## Aggregate Summary\n\n")
                f.write(
                    "- This is an instruction-based LLM aggregation of the previous 'summary batches'.\n"
                )
                f.write(f"- Aggregation model: {aggregator.model_name}\n\n")
                f.write(f"{full_summary}\n\n")
            logging.info(f"Updated {out_path} with aggregate summary")
        except Exception as e:
            logging.error(f"unable to update {out_path} with aggregate summary: {e}")

    full_summary_html = f"""
        <div style="
            margin-bottom: 20px;
            font-size: 18px;
            line-height: 1.5em;
            color: #333;
        ">
            <h2 style="font-size: 22px; color: #555;">Aggregate Summary:</h2>
            <p style="white-space: pre-line;">{full_summary}</p>
        </div>
        """
    return full_summary_html


def predict(
    input_text: str,
    model_name: str,
    token_batch_length: int = 1024,
    empty_cache: bool = True,
    **settings,
) -> list:
    """
    predict - helper fn to support multiple models for summarization at once

    :param str input_text: the input text to summarize
    :param str model_name: model name to use
    :param int token_batch_length: the length of the token batches to use
    :param bool empty_cache: whether to empty the cache before loading a new= model
    :return: list of dicts with keys "summary" and "score"
    """
    if torch.cuda.is_available() and empty_cache:
        torch.cuda.empty_cache()

    model, tokenizer = load_model_and_tokenizer(model_name)
    summaries = summarize_via_tokenbatches(
        input_text,
        model,
        tokenizer,
        batch_length=token_batch_length,
        **settings,
    )

    del model
    del tokenizer
    gc.collect()

    return summaries


def proc_submission(
    input_text: str,
    model_name: str,
    num_beams: int,
    token_batch_length: int,
    length_penalty: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    predrop_stopwords: bool,
    max_input_length: int = 6144,
):
    """
    proc_submission - a helper function for the gradio module to process submissions

    Args:
        input_text (str): the input text to summarize
        model_name (str): the hf model tag of the model to use
        num_beams (int): the number of beams to use
        token_batch_length (int): the length of the token batches to use
        length_penalty (float): the length penalty to use
        repetition_penalty (float): the repetition penalty to use
        no_repeat_ngram_size (int): the no repeat ngram size to use
        predrop_stopwords (bool): whether to pre-drop stopwords before truncating/summarizing
        max_input_length (int, optional): the maximum input length to use. Defaults to 6144.

    Note:
        the max_input_length is set to 6144 by default, but can be changed by setting the
        environment variable APP_MAX_WORDS to a different value.

    Returns:
        tuple (4): a tuple containing the following:
    """

    remove_stagnant_files()  # clean up old files
    settings = {
        "length_penalty": float(length_penalty),
        "repetition_penalty": float(repetition_penalty),
        "no_repeat_ngram_size": int(no_repeat_ngram_size),
        "encoder_no_repeat_ngram_size": 4,
        "num_beams": int(num_beams),
        "min_length": 4,
        "max_length": int(token_batch_length // 4),
        "early_stopping": True,
        "do_sample": False,
    }
    max_input_length = int(os.environ.get("APP_MAX_WORDS", max_input_length))
    logging.info(
        f"max_input_length set to: {max_input_length}. pre-drop stopwords: {predrop_stopwords}"
    )

    st = time.perf_counter()
    history = {}
    cln_text = clean(input_text, lower=False)
    parsed_cln_text = remove_stopwords(cln_text) if predrop_stopwords else cln_text
    logging.info(
        f"pre-truncation word count: {len(contraction_aware_tokenize(parsed_cln_text))}"
    )
    truncation_validated = truncate_word_count(
        parsed_cln_text, max_words=max_input_length
    )

    if truncation_validated["was_truncated"]:
        model_input_text = truncation_validated["processed_text"]
        # create elaborate HTML warning
        input_wc = len(contraction_aware_tokenize(parsed_cln_text))
        msg = f"""
        <div style="background-color: #FFA500; color: white; padding: 20px;">
        <h3>Warning</h3>
        <p>Input text was truncated to {max_input_length} words. That's about {100*max_input_length/input_wc:.2f}% of the original text.</p>
        <p>Dropping stopwords is set to {predrop_stopwords}. If this is not what you intended, please validate the advanced settings.</p>
        </div>
        """
        logging.warning(msg)
        history["WARNING"] = msg
    else:
        model_input_text = truncation_validated["processed_text"]
        msg = None

    if len(input_text) < 50:
        # this is essentially a different case from the above
        msg = f"""
        <div style="background-color: #880808; color: white; padding: 20px;">
        <br>
        <img src="https://i.imgflip.com/7kadd9.jpg" alt="no text">
        <br>
        <h3>Error</h3>
        <p>Input text is too short to summarize. Detected {len(input_text)} characters.
        Please load text by selecting an example from the dropdown menu or by pasting text into the text box.</p>
        </div>
        """
        logging.warning(msg)
        logging.warning("RETURNING EMPTY STRING")
        history["WARNING"] = msg

        return msg, "<strong>No summary generated.</strong>", "", []

    _summaries = predict(
        input_text=model_input_text,
        model_name=model_name,
        token_batch_length=token_batch_length,
        **settings,
    )
    sum_text = [s["summary"][0].strip() + "\n" for s in _summaries]
    sum_scores = [
        f" - Batch Summary {i}: {round(s['summary_score'],4)}"
        for i, s in enumerate(_summaries)
    ]

    full_summary = textlist2html(sum_text)
    history["Summary Scores"] = "<br><br>"
    scores_out = "\n".join(sum_scores)
    rt = round((time.perf_counter() - st) / 60, 2)
    logging.info(f"Runtime: {rt} minutes")
    html = ""
    html += f"<p>Runtime: {rt} minutes with model: {model_name}</p>"
    if msg is not None:
        html += msg

    html += ""

    settings["remove_stopwords"] = predrop_stopwords
    settings["model_name"] = model_name
    saved_file = saves_summary(summarize_output=_summaries, outpath=None, **settings)
    return html, full_summary, scores_out, saved_file


def load_single_example_text(
    example_path: str or Path,
    max_pages: int = 20,
) -> str:
    """
    load_single_example_text - loads a single example text file

    :param strorPath example_path: name of the example to load
    :param int max_pages: the maximum number of pages to load from a PDF
    :return str: the text of the example
    """
    global name_to_path, ocr_model
    full_ex_path = name_to_path[example_path]
    full_ex_path = Path(full_ex_path)
    if full_ex_path.suffix in [".txt", ".md"]:
        with open(full_ex_path, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()
        text = clean(raw_text, lower=False)
    elif full_ex_path.suffix == ".pdf":
        logging.info(f"Loading PDF file {full_ex_path}")
        max_pages = int(os.environ.get("APP_OCR_MAX_PAGES", max_pages))
        logging.info(f"max_pages set to: {max_pages}")
        conversion_stats = convert_PDF_to_Text(
            full_ex_path,
            ocr_model=ocr_model,
            max_pages=max_pages,
        )
        text = conversion_stats["converted_text"]
    else:
        logging.error(f"Unknown file type {full_ex_path.suffix}")
        text = "ERROR - check example path"

    return text


def load_uploaded_file(file_obj, max_pages: int = 20, lower: bool = False) -> str:
    """
    load_uploaded_file - loads a file uploaded by the user

    :param file_obj (POTENTIALLY list): Gradio file object inside a list
    :param int max_pages: the maximum number of pages to load from a PDF
    :param bool lower: whether to lowercase the text
    :return str: the text of the file
    """
    global ocr_model
    logger = logging.getLogger(__name__)
    # check if mysterious file object is a list
    if isinstance(file_obj, list):
        file_obj = file_obj[0]
    file_path = Path(file_obj.name)
    try:
        logger.info(f"Loading file:\t{file_path}")
        if file_path.suffix in [".txt", ".md"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
            text = clean(raw_text, lower=lower)
        elif file_path.suffix == ".pdf":
            logger.info(f"loading a PDF file: {file_path.name}")
            max_pages = int(os.environ.get("APP_OCR_MAX_PAGES", max_pages))
            logger.info(f"max_pages is: {max_pages}. Starting conversion...")
            conversion_stats = convert_PDF_to_Text(
                file_path,
                ocr_model=ocr_model,
                max_pages=max_pages,
            )
            text = conversion_stats["converted_text"]
        else:
            logger.error(f"Unknown file type:\t{file_path.suffix}")
            text = "ERROR - check file - unknown file type. PDF, TXT, and MD are supported."

        return text
    except Exception as e:
        logger.error(f"Trying to load file:\t{file_path},\nerror:\t{e}")
        return f"Error: Could not read file {file_path.name}. Make sure it is a PDF, TXT, or MD file."


def parse_args():
    """arguments for the command line interface"""
    parser = argparse.ArgumentParser(
        description="Document Summarization with Long-Document Transformers - Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Runs a local-only web UI to summarize documents. pass --share for a public link to share.",
    )

    parser.add_argument(
        "--share",
        dest="share",
        action="store_true",
        help="Create a public link to share",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        help=f"Add a custom model to the list of models: {pp.pformat(MODEL_OPTIONS, compact=True)}",
    )
    parser.add_argument(
        "-nb",
        "--add_beam_option",
        type=int,
        default=None,
        help=f"Add a beam search option to the demo UI options, default: {pp.pformat(BEAM_OPTIONS, compact=True)}",
    )
    parser.add_argument(
        "-batch",
        "--token_batch_option",
        type=int,
        default=None,
        help=f"Add a token batch size to the demo UI options, default: {pp.pformat(TOKEN_BATCH_OPTIONS, compact=True)}",
    )
    parser.add_argument(
        "-max_agg",
        "-2x",
        "--aggregator_beam_boost",
        dest="aggregator_beam_boost",
        action="store_true",
        help="Double the number of beams for the aggregator during beam search",
    )
    parser.add_argument(
        "-level",
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level",
    )

    return parser.parse_args()


if __name__ == "__main__":
    """main - the main function of the app"""
    logger = logging.getLogger(__name__)
    args = parse_args()
    logger.setLevel(args.log_level)
    logger.info(f"args: {pp.pformat(args.__dict__, compact=True)}")

    # add any custom options
    if args.model is not None:
        logger.info(f"Adding model {args.model} to the list of models")
        MODEL_OPTIONS.append(args.model)
    if args.add_beam_option is not None:
        logger.info(f"Adding beam search option {args.add_beam_option} to the list")
        BEAM_OPTIONS.append(args.add_beam_option)
    if args.token_batch_option is not None:
        logger.info(f"Adding token batch option {args.token_batch_option} to the list")
        TOKEN_BATCH_OPTIONS.append(args.token_batch_option)

    if args.aggregator_beam_boost:
        logger.info("Doubling aggregator num_beams")
        _agg_cfg = aggregator.get_generation_config()
        _agg_cfg["num_beams"] = _agg_cfg["num_beams"] * 2
        aggregator.update_generation_config(**_agg_cfg)

    logger.info("Loading OCR model")
    with contextlib.redirect_stdout(None):
        ocr_model = ocr_predictor(
            "db_resnet50",
            "crnn_mobilenet_v3_large",
            pretrained=True,
            assume_straight_pages=True,
        )

    # load the examples
    name_to_path = load_example_filenames(_here / "examples")
    logger.info(f"Loaded {len(name_to_path)} examples")

    demo = gr.Blocks(title="Document Summarization with Long-Document Transformers")
    _examples = list(name_to_path.keys())
    logger.info("Starting app instance")
    with demo:
        gr.Markdown("# Document Summarization with Long-Document Transformers")
        gr.Markdown(
            """An example use case for fine-tuned long document transformers. Model(s) are trained on [book summaries](https://hf.co/datasets/kmfoda/booksum). Architectures [in this demo](https://hf.co/spaces/pszemraj/document-summarization) are [LongT5-base](https://hf.co/pszemraj/long-t5-tglobal-base-16384-book-summary) and [Pegasus-X-Large](https://hf.co/pszemraj/pegasus-x-large-book-summary).

            **Want more performance? Run this demo from a free Google Colab GPU:**.
            <br>
            <a href="https://colab.research.google.com/gist/pszemraj/52f67cf7326e780155812a6a1f9bb724/document-summarization-on-gpu.ipynb">
            <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
            </a>
            <br>
            """
        )
        with gr.Column():
            gr.Markdown("## Load Inputs & Select Parameters")
            gr.Markdown(
                """Enter/paste text below, or upload a file. Pick a model & adjust params (_optional_), and press **Summarize!**

                See [the guide doc](https://gist.github.com/pszemraj/722a7ba443aa3a671b02d87038375519) for details.
                """
            )
            with gr.Row(variant="compact"):
                with gr.Column(scale=0.5, variant="compact"):
                    model_name = gr.Dropdown(
                        choices=MODEL_OPTIONS,
                        value=MODEL_OPTIONS[0],
                        label="Model Name",
                    )
                    num_beams = gr.Radio(
                        choices=BEAM_OPTIONS,
                        value=BEAM_OPTIONS[len(BEAM_OPTIONS) // 2],
                        label="Beam Search: # of Beams",
                    )
                    load_examples_button = gr.Button(
                        "Load Example in Dropdown",
                    )
                    load_file_button = gr.Button("Upload & Process File")
                with gr.Column(variant="compact"):
                    example_name = gr.Dropdown(
                        _examples,
                        label="Examples",
                        value=random.choice(_examples),
                    )
                    uploaded_file = gr.File(
                        label="File Upload",
                        file_count="single",
                        file_types=[".txt", ".md", ".pdf"],
                        type="file",
                    )
            with gr.Row():
                input_text = gr.Textbox(
                    lines=4,
                    max_lines=12,
                    label="Text to Summarize",
                    placeholder="Enter text to summarize, the text will be cleaned and truncated on Spaces. Narrative, academic (both papers and lecture transcription), and article text work well. May take a bit to generate depending on the input text :)",
                )
        gr.Markdown("---")
        with gr.Column():
            gr.Markdown("## Generate Summary")
            with gr.Row():
                summarize_button = gr.Button(
                    "Summarize!",
                    variant="primary",
                )
                gr.Markdown(
                    "_Summarization should take ~1-2 minutes for most settings, but may extend up to 5-10 minutes in some scenarios._"
                )
            output_text = gr.HTML("<p><em>Output will appear below:</em></p>")
            with gr.Column():
                gr.Markdown("### Results & Scores")
                with gr.Row():
                    with gr.Column(variant="compact"):
                        gr.Markdown(
                            "Download the summary as a text file, with parameters and scores."
                        )
                        text_file = gr.File(
                            label="Download as Text File",
                            file_count="single",
                            type="file",
                            interactive=False,
                        )
                    with gr.Column(variant="compact"):
                        gr.Markdown(
                            "Scores **roughly** represent the summary quality as a measure of the model's 'confidence'. less-negative numbers (closer to 0) are better."
                        )
                        summary_scores = gr.Textbox(
                            label="Summary Scores",
                            placeholder="Summary scores will appear here",
                        )
            with gr.Column(variant="panel"):
                gr.Markdown("### **Summary Output**")
                summary_text = gr.HTML(
                    label="Summary",
                    value="<center><i>Summary will appear here!</i></center>",
                )
            with gr.Column():
                gr.Markdown("### **Aggregate Summary Batches**")
                gr.Markdown(
                    "_Note: this is an experimental feature. Feedback welcome in the [discussions](https://hf.co/spaces/pszemraj/document-summarization/discussions)!_"
                )
                with gr.Row():
                    aggregate_button = gr.Button(
                        "Aggregate!",
                        variant="primary",
                    )
                    gr.Markdown(
                        f"""Aggregate the above batches into a cohesive summary.
                    - A secondary instruct-tuned LM consolidates info
                    - Current model: [{AGGREGATE_MODEL}](https://hf.co/{AGGREGATE_MODEL})
                                """
                    )
                with gr.Column(variant="panel"):
                    aggregated_summary = gr.HTML(
                        label="Aggregate Summary",
                        value="<center><i>Aggregate summary will appear here!</i></center>",
                    )
                    gr.Markdown(
                        "\n\n_Aggregate summary is also appended to the bottom of the `.txt` file._"
                    )

        gr.Markdown("---")
        with gr.Column():
            gr.Markdown("### Advanced Settings")
            gr.Markdown(
                "Refer to [the guide doc](https://gist.github.com/pszemraj/722a7ba443aa3a671b02d87038375519) for what these are, and how they impact _quality_ and _speed_."
            )
            with gr.Row(variant="compact"):
                length_penalty = gr.Slider(
                    minimum=0.3,
                    maximum=1.1,
                    label="length penalty",
                    value=0.7,
                    step=0.05,
                )
                token_batch_length = gr.Radio(
                    choices=TOKEN_BATCH_OPTIONS,
                    label="token batch length",
                    # select median option
                    value=TOKEN_BATCH_OPTIONS[len(TOKEN_BATCH_OPTIONS) // 2],
                )

            with gr.Row(variant="compact"):
                repetition_penalty = gr.Slider(
                    minimum=1.0,
                    maximum=5.0,
                    label="repetition penalty",
                    value=1.5,
                    step=0.1,
                )
                no_repeat_ngram_size = gr.Radio(
                    choices=[2, 3, 4, 5],
                    label="no repeat ngram size",
                    value=3,
                )
                predrop_stopwords = gr.Checkbox(
                    label="Drop Stopwords (Pre-Truncation)",
                    value=False,
                )
        with gr.Column():
            gr.Markdown("## About")
            gr.Markdown(
                "- Models are fine-tuned on the [🅱️ookSum dataset](https://arxiv.org/abs/2105.08209). The goal was to create a model that generalizes well and is useful for summarizing text in academic and everyday use."
            )
            gr.Markdown(
                "- _Update April 2023:_ Additional models fine-tuned on the [PLOS](https://hf.co/datasets/pszemraj/scientific_lay_summarisation-plos-norm) and [ELIFE](https://hf.co/datasets/pszemraj/scientific_lay_summarisation-elife-norm) subsets of the [scientific lay summaries](https://arxiv.org/abs/2210.09932) dataset are available (see dropdown at the top)."
            )
            gr.Markdown(
                "Adjust the max input words & max PDF pages for OCR by duplicating this space and [setting the environment variables](https://hf.co/docs/hub/spaces-overview#managing-secrets) `APP_MAX_WORDS` and `APP_OCR_MAX_PAGES` to the desired integer values."
            )
            gr.Markdown("---")

        load_examples_button.click(
            fn=load_single_example_text, inputs=[example_name], outputs=[input_text]
        )

        load_file_button.click(
            fn=load_uploaded_file, inputs=uploaded_file, outputs=[input_text]
        )

        summarize_button.click(
            fn=proc_submission,
            inputs=[
                input_text,
                model_name,
                num_beams,
                token_batch_length,
                length_penalty,
                repetition_penalty,
                no_repeat_ngram_size,
                predrop_stopwords,
            ],
            outputs=[output_text, summary_text, summary_scores, text_file],
        )
        aggregate_button.click(
            fn=aggregate_text,
            inputs=[summary_text, text_file],
            outputs=[aggregated_summary],
        )
    demo.launch(enable_queue=True, share=args.share)
