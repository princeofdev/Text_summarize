"""
    utils.py - Utility functions for the project.
"""
import logging
import os
import re
import string
import subprocess
from collections import defaultdict, deque
from datetime import datetime, timedelta
from itertools import combinations, islice
from pathlib import Path
from typing import List

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)

import torch
from natsort import natsorted
from nltk.tokenize import WhitespaceTokenizer, sent_tokenize, word_tokenize
from rapidfuzz import fuzz

STOPWORDS = set(
    "a about above after again all also am an and any are aren't as at back be because been before being below between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had hadn't has hasn't have haven't having he'd he'll he's hence her here here's hers herself him himself his how how's however i'd i'll i'm i've if in into is isn't it's its itself just let's me more moreover most mustn't my myself new nor now of off on once only or other ought our ours ourselves out over own really same shan't she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's therefore these they they'd they'll they're they've this those through thus to too under until up use used using very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's with won't would wouldn't you'd you'll you're you've your yours yourself yourselves".split()
)


def contraction_aware_tokenize(text: str) -> List[str]:
    """contraction_aware_tokenize - merges words containing apostrophes as one token."""

    # Tokenize the text using the WhitespaceTokenizer
    tokenizer = WhitespaceTokenizer()
    tokens = tokenizer.tokenize(text)

    merged_tokens = []
    merged_token = ""

    for token in tokens:
        if re.search(r"\w+'\w+", token):
            # Token contains an apostrophe, merge with previous token
            merged_token += token
        else:
            # no apostrophe, add previous merged token (if any) and current
            if merged_token:
                merged_tokens.append(merged_token)
                merged_token = ""
            merged_tokens.append(token)

    # Add the last merged token (if any)
    if merged_token:
        merged_tokens.append(merged_token)

    return merged_tokens


def remove_stopwords(
    text: str, stopwords: List[str] = STOPWORDS, contraction_tokenize: bool = True
) -> str:
    """
    remove_stopwords - Remove stopwords from text.

    :param str text: input text
    :param List[str] stopwords: list of stopwords, defaults to STOPWORDS
    :param bool contraction_tokenize: use custom apostrophe tokenizer, defaults to True
    :return str: text with stopwords removed
    """
    lines = text.split("\n")
    filtered_lines = []

    def fix_commas(text: str) -> str:
        """fixes commas in text to have a space after them"""
        spaced_text = text.replace(",", ", ")
        return spaced_text.replace("  ", " ").strip()

    for line in lines:
        sentences = sent_tokenize(line)
        filtered_sentences = []

        for sentence in sentences:
            # Add space around punctuations for the regex to work correctly, only if they are followed by a letter
            sentence_with_spaces = re.sub(r"([.,!?])(\w)", r"\1 \2", sentence[:-1])

            words = (
                contraction_aware_tokenize(sentence_with_spaces)
                if contraction_tokenize
                else word_tokenize(sentence_with_spaces)
            )

            filtered_words = []
            for word in words:
                if word.lower() not in stopwords:
                    filtered_words.append(word)

            filtered_sentence = " ".join(filtered_words)
            # Restore original spaces around punctuation marks
            filtered_sentence = re.sub(r"([.,!?])\s*", r"\1", filtered_sentence)

            filtered_sentences.append(filtered_sentence + sentence[-1])

        filtered_line = " ".join(filtered_sentences)

        # Replace multiple consecutive whitespaces with a single space
        filtered_line = re.sub(r"\s+", " ", filtered_line)
        filtered_line = fix_commas(filtered_line.strip())

        filtered_lines.append(filtered_line)

    filtered_text = "\n".join(filtered_lines)

    return filtered_text


def remove_stagnant_files(
    freq: str = "hourly",
    search_path: str = ".",
    substring="DocSumm",
    remove_suffix=".txt",
):
    """
    remove_stagnant_files - Remove files that have not been modified in a certain amount of time.

    :param str freq: frequency of file removal, defaults to "hourly"
    :param str search_path: location to search for files, defaults to "."
    :param str substring: substring to search for in file names, defaults to "DocSumm"
    :param str remove_suffix: suffix of files to remove, defaults to ".txt"
    :raises ValueError: if freq is not one of "hourly", "daily", or "weekly"
    """
    current_time = datetime.now()
    search_path = Path(search_path)

    if freq == "hourly":
        time_threshold = current_time - timedelta(hours=1)
    elif freq == "daily":
        time_threshold = current_time - timedelta(days=1)
    elif freq == "weekly":
        time_threshold = current_time - timedelta(weeks=1)
    else:
        raise ValueError(
            "Invalid frequency. Supported values are 'hourly', 'daily', and 'weekly'."
        )

    files_to_remove = []
    potential_files = [
        f for f in search_path.iterdir() if f.is_file() and f.suffix == remove_suffix
    ]
    logging.info(f"Found {len(potential_files)} files.")
    for candidate in potential_files:
        if (
            candidate.is_file()
            and substring in candidate.name
            and candidate.stat().st_mtime < time_threshold.timestamp()
        ):
            files_to_remove.append(candidate)
        logging.debug(f"File {candidate} last modified at {candidate.stat().st_mtime}")
    logging.info(f"Removing {len(files_to_remove)} files.")
    for file_path in files_to_remove:
        file_path.unlink()
    logging.debug(f"Removed files: {files_to_remove}")


def compare_model_size(model_name: str, threshold: int = 500) -> bool:
    """
    compare_model_size - compare string representations of model size to a threshold

    :param str model_name: the model name to compare
    :param int threshold: the threshold to compare against in millions, defaults to 500
    :return: True if the model size is greater than the threshold, False or None otherwise
    """
    pattern = r"(\d+)(M|G|k|b)?"  # param regex

    matches = re.findall(pattern, model_name)
    if not matches:
        return None

    # Extract the parameter count and unit
    parameter_count, unit = matches[-1]
    parameter_count = int(parameter_count)

    # Convert to the standard form (M for million, G for billion, k for thousand)
    if unit == "G" or unit == "b":
        parameter_count *= 1000
    elif unit == "M":
        pass
    elif unit == "k":
        parameter_count /= 1000
    else:
        return None  # Unknown

    return parameter_count > threshold


def validate_pytorch2(torch_version: str = None) -> bool:
    """
    validate_pytorch2 - validate that the PyTorch version is 2.0 or greater

    :param str torch_version: the PyTorch version to validate, defaults to None
    :return: True if the PyTorch version is 2.0 or greater, False otherwise
    """

    torch_version = torch.__version__ if torch_version is None else torch_version

    pattern = r"^2\.\d+(\.\d+)*"

    return True if re.match(pattern, torch_version) else False


def get_timestamp(detailed=False) -> str:
    """
    get_timestamp - get a timestamp for the current time
    :param bool detailed: whether to include seconds and microseconds, defaults to False
    :return: str, the timestamp
    """
    return (
        datetime.now().strftime("%b%d%Y_%H%M%S%f")
        if detailed
        else datetime.now().strftime("%b%d%Y_%H")
    )


def truncate_word_count(text: str, max_words=1024) -> dict:
    """
    truncate_word_count - truncate a text to a maximum number of words
    :param str text: the text to truncate
    :param int max_words: the maximum number of words to keep, defaults to 1024
    :return: dict, the processed text
    """
    words = contraction_aware_tokenize(str(text))
    processed = {}
    if len(words) > max_words:
        processed["was_truncated"] = True
        processed["processed_text"] = " ".join(words[:max_words])
    else:
        processed["was_truncated"] = False
        processed["processed_text"] = text
    return processed


def load_examples(src, filetypes=[".txt", ".pdf"]):
    """
    load_examples - a helper function for the gradio module to load examples
    :param str src: the path to the examples
    """
    src = Path(src)
    src.mkdir(exist_ok=True)

    pdf_url = (
        "https://www.dropbox.com/s/y92xy7o5qb88yij/all_you_need_is_attention.pdf?dl=1"
    )
    subprocess.run(["wget", pdf_url, "-O", src / "all_you_need_is_attention.pdf"])
    examples = [f for f in src.iterdir() if f.suffix in filetypes]
    examples = natsorted(examples)
    # load the examples into a list
    text_examples = []
    for example in examples:
        with open(example, "r") as f:
            text = f.read()
            text_examples.append([text, "base", 2, 1024, 0.7, 3.5, 3])

    return text_examples


def load_example_filenames(example_path: str or Path):
    """
    load_example_filenames - a helper function for the gradio module to load examples
    Returns:
        dict, the examples (filename:full path)
    """
    example_path = Path(example_path)
    # load the examples into a list
    examples = {f.name: f for f in example_path.glob("*.txt")}
    return examples


def textlist2html(text_batches: List[str]) -> str:
    """textlist2html - convert a list of text summaries into a single HTML string"""
    # Step 1: Generate each summary batch as a string of HTML
    formatted_batches = [
        f"""
        <div style="
            margin-bottom: 20px;
            font-size: 18px;
            line-height: 1.5em;
            color: #333;
        ">
            <h2 style="font-size: 22px; color: #555;">Batch {i}:</h2>
            <p style="white-space: pre-line;">{s}</p>
        </div>
        """
        for i, s in enumerate(text_batches, start=1)
    ]

    # Step 2: Join all the summary batches together into one string
    joined_batches = "".join(formatted_batches)

    # Step 3: Wrap the summary string in a larger div with background color, border, and padding
    text_html_block = f"""
    <div style="
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 20px;
    ">
    {joined_batches}
    </div>
    """

    return text_html_block


def extract_batches(html_string: str, pattern=None, flags=None) -> list:
    """
    Extract batches of text from an HTML string.

    Args:
        html_string (str): The HTML string to extract batches from.
        pattern (str, optional): The regular expression pattern to use. Defaults to a pattern that matches batches in the format provided.
        flags (int, optional): The flags to use with the regular expression. Defaults to re.DOTALL.

    Returns:
        list: A list of dictionaries where each dictionary represents a batch and has 'title' and 'content' keys.
    """
    # Set default pattern if none provided
    if pattern is None:
        pattern = r'<h2 style="font-size: 22px; color: #555;">(.*?)</h2>\s*<p style="white-space: pre-line;">(.*?)</p>'

    # Set default flags if none provided
    if flags is None:
        flags = re.DOTALL

    try:
        # Find all matches in the string
        matches = re.findall(pattern, html_string, flags)

        # Convert matches to a list of dictionaries
        batches = [
            {"title": title.strip(), "content": content.strip()}
            for title, content in matches
        ]

        return batches
    except re.error as e:
        logging.error(f"An error occurred while trying to extract batches: {e}")
        return []


def extract_keywords(
    text: str, num_keywords: int = 3, window_size: int = 5, kw_max_len: int = 20
) -> List[str]:
    """
    Extracts keywords from a text using a simplified TextRank algorithm.

    Args:
        text: The text to extract keywords from.
        num_keywords: The number of keywords to extract. Default: 3
        window_size: The number of words considered for co-occurrence. Default: 5
        kw_max_len: The maximum length of a keyword (truncate longer keywords to max). Default: 20
    Returns:
        A list of strings, where each string is a keyword extracted from the input text.
    """
    logger = logging.getLogger(__name__)
    # Remove stopwords and tokenize the text into words
    words = [
        word
        for word in re.findall(r"\b\w{3,}\b", text.lower())
        if word not in STOPWORDS
    ]

    # Create a graph of word co-occurrences within a moving window of words
    cooccur = defaultdict(lambda: defaultdict(int))
    deque_words = deque(maxlen=window_size)
    for word in words:
        for w1, w2 in combinations(deque_words, 2):
            cooccur[w1][w2] += 1
            cooccur[w2][w1] += 1
        deque_words.append(word)

    # Assign scores to words using a simplified TextRank algorithm
    scores = defaultdict(float)
    for _ in range(10):
        new_scores = defaultdict(float)
        for word, co_words in cooccur.items():
            new_scores[word] = 0.15 + 0.85 * sum(
                cooccur[word][other] / sum(cooccur[other].values()) * scores[other]
                for other in co_words
            )
        scores = new_scores

    # Sort the words by score and return the top num_keywords keywords
    keywords = sorted(scores, key=scores.get, reverse=True)[:num_keywords]
    logger.debug(f"All keywords: {keywords}")
    # Use fuzzy matching to remove similar keywords
    final_keywords = []
    for keyword in keywords:
        if not any(fuzz.ratio(keyword, other) > 70 for other in final_keywords):
            final_keywords.append(keyword[:kw_max_len])
    logger.debug(f"Keywords (max len. {kw_max_len}):\t{final_keywords}")
    return final_keywords


def saves_summary(
    summarize_output, outpath: str or Path = None, add_signature=True, **kwargs
) -> Path:
    """
    saves_summary - save the summary generated from summarize_via_tokenbatches() to a text file

    summarize_output: output from summarize_via_tokenbatches()
    outpath: path to the output file
    add_signature: whether to add a signature to the output file
    kwargs: additional keyword arguments to include in the output file
    """
    logger = logging.getLogger(__name__)
    sum_text = [f"{s['summary'][0]}\n" for s in summarize_output]
    sum_scores = [f"\n - {round(s['summary_score'],4)}" for s in summarize_output]
    scores_text = "\n".join(sum_scores)
    full_summary = "\n".join(sum_text)

    keywords = "_".join(extract_keywords(full_summary, kw_max_len=4))
    logger.debug(f"kw:\t{keywords}")
    outpath = (
        Path.cwd() / f"DocSumm_{keywords}_{get_timestamp()}.txt"
        if outpath is None
        else Path(outpath)
    )
    logger.info(f"Saving summary to:\t{outpath.name}")
    with open(
        outpath,
        "w",
        encoding="utf-8",
    ) as fo:
        fo.writelines(full_summary)
        fo.write("\n\n")
        if add_signature:
            fo.write("\n\n---\n\n")
            fo.write("Generated with the Document Summarization space :)\n\n")
            fo.write("https://hf.co/spaces/pszemraj/document-summarization\n\n")
    with open(
        outpath,
        "a",
        encoding="utf-8",
    ) as fo:
        fo.write("\n")
        fo.write(f"## Section Scores:\n\n")
        fo.writelines(scores_text)
        fo.write("\n\n")
        fo.write(f"Date: {get_timestamp()}\n\n")
        if kwargs:
            fo.write("---\n\n")
            fo.write("## Parameters:\n\n")
            for key, value in kwargs.items():
                fo.write(f"{key}: {value}\n")
    return outpath
