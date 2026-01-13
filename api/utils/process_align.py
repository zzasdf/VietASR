import requests as r
from typing import List, Tuple, Dict, Any


API_TEXT_NORM = 'https://speech.aiservice.vn/asr/textnorm'


def call_text_norm(text: str, domain: str = None) -> Dict[str, Any]:
    """
    Call the text normalization API to normalize the input text.

    Args:
        text (str): The input text to be normalized.
        domain (str, optional): The domain/context for normalization. Defaults to None.

    Returns:
        dict: The API response containing normalized text and replacement dictionary.

    Raises:
        requests.exceptions.RequestException: If the API request fails.
    """
    response = r.post(API_TEXT_NORM, json={"text": text, "domain": domain})
    return response.json()


def _find_start_indices(main_list: List[str], sublist: List[str]) -> List[int]:
    """
    Find all starting indices where sublist appears in main_list.

    Args:
        main_list (List[str]): The main list to search in.
        sublist (List[str]): The sublist to search for.

    Returns:
        List[int]: List of starting indices where sublist is found.
    """
    start_indices = []
    try:
        index = main_list.index(sublist[0])
        while index <= len(main_list) - len(sublist):
            if main_list[index:index + len(sublist)] == sublist:
                start_indices.append(index)
            index = main_list.index(sublist[0], index + 1)
    except ValueError:
        pass
    return start_indices


def process_text_norm(
    text: str,
    segments: List[Tuple[float, float, str]],
    domain: str = None
) -> Tuple[str, List[Tuple[float, float, str]]]:
    """
    Process text normalization and adjust the corresponding segments.

    Args:
        text (str): The original text to be normalized.
        segments (List[Tuple[float, float, str]]): List of segments with start time,
            end time, and corresponding word.
        domain (str, optional): The domain/context for normalization. Defaults to None.

    Returns:
        Tuple[str, List[Tuple[float, float, str]]]: A tuple containing:
            - The normalized text
            - The adjusted segments with normalized words and proper timings

    Example:
        >>> text = 'xin chào phao đây sừn ba a ma zôn'
        >>> segments = [(0.1, 0.2, 'xin'), (0.2, 0.3, 'chào'), ...]
        >>> normalized_text, normalized_segments = process_text_norm(text, segments)
    """
    # Call text normalization API
    norm_result = call_text_norm(text, domain=domain)
    out_text = norm_result["result"]["text"]
    replace_dict = norm_result["result"]["replace_dict"]

    # Find all replacement positions in the original text
    replace_obj = {}
    list_starts = []
    text_words = text.split()

    for original_phrase, normalized_phrase in replace_dict.items():
        start_indices = _find_start_indices(text_words, original_phrase.split())
        list_starts += start_indices

        for start_idx in start_indices:
            replace_obj[start_idx] = {
                "org_text": original_phrase,
                "norm_text": normalized_phrase,
                "end_idx": start_idx + len(original_phrase.split())
            }

    # Sort the starting indices for processing in order
    list_starts = sorted(list_starts)

    # Reconstruct segments with normalized text
    out_segments = []
    i = 0
    n = len(segments)

    while i < n:
        if i in list_starts:
            # Handle replacement case
            replacement = replace_obj[i]
            start_time = segments[i][0]
            end_time = segments[replacement["end_idx"] - 1][1]
            out_segments.append([start_time, end_time, replacement["norm_text"]])
            i = replacement["end_idx"]  # Skip the replaced words
        else:
            # Handle normal case
            start, end, word = segments[i]
            out_segments.append([start, end, word])
            i += 1

    return out_text, out_segments


if __name__ == "__main__":
    # Example usage
    text = 'xin chào phao đây sừn ba a ma zôn phao đây sừn hehe cha vồ'
    segments = []
    start = 0.0
    for word in text.split():
        start += 0.1
        end = start + 0.1
        segments.append((start, end, word))

    normalized_text, normalized_segments = process_text_norm(text, segments)

    print("Original Text:", text)
    print("Normalized Text:", normalized_text)
    print("Segments:")
    for seg in normalized_segments:
        print(f"{seg[0]:.1f}-{seg[1]:.1f}s: {seg[2]}")
