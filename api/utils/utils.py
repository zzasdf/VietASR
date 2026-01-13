def merge_short_audio_segments(segments, target_length):
    """
    Groups short audio segments into sublists where the combined duration is closest to the target length.

    The algorithm merges consecutive segments while keeping the cumulative duration as close as possible
    to the target length. When adding a new segment would take the cumulative duration further from
    the target than the current duration, a new group is started.

    Args:
        segments: List of audio segments, where each segment is represented as a (start, end) tuple
        target_length: The desired duration for each merged group of segments

    Returns:
        A list of segment groups, where each group is a list of (start, end) tuples
        with combined duration close to the target length
    """
    # Handle empty input case
    if not segments:
        return []

    result = []
    current_sublist = []      # Current group of segments being built
    current_length = 0        # Cumulative duration of current_sublist

    for segment in segments:
        start, end = segment
        segment_length = end - start
        new_length = current_length + segment_length

        # Decision point: whether to add to current group or start a new one
        if (current_sublist  # Only consider splitting if we have segments already
            # Check if adding this segment would take us further from target
                and abs(new_length - target_length) > abs(current_length - target_length)):

            # Finalize current group and start a new one
            result.append(current_sublist)
            current_sublist = [segment]
            current_length = segment_length
        else:
            # Add segment to current group
            current_sublist.append(segment)
            current_length = new_length

    # Add the last group if it contains any segments
    if current_sublist:
        result.append(current_sublist)

    return result
