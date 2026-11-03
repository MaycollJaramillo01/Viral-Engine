"""Read SRT cue boundaries before adapting them to MoviePy timing tuples."""
import os
import re


def file_to_subtitles(filename):
    if not filename or not os.path.isfile(filename):
        return []

    times_texts = []
    current_times = None
    current_text = ""
    index = 0
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            times = re.findall("([0-9]*:[0-9]*:[0-9]*,[0-9]*)", line)
            # Once a cue has started, timestamps belong to its text until the
            # blank-line separator; they must not overwrite the cue's timing.
            if times and current_times is None:
                current_times = line
            elif line.strip() == "" and current_times:
                index += 1
                times_texts.append((index, current_times.strip(), current_text.strip()))
                current_times, current_text = None, ""
            elif current_times:
                current_text += line

    # Flush the final block. SRT files whose last subtitle is not followed by a
    # trailing blank line never hit the blank-line branch above, so without this
    # the last subtitle would be silently dropped.
    if current_times:
        index += 1
        times_texts.append((index, current_times.strip(), current_text.strip()))
    return times_texts


def moviepy_subtitles(filename):
    """Convert only cue headers; timestamp-bearing text remains caption text."""
    from moviepy.tools import convert_to_seconds

    cues = []
    for _index, timing, text in file_to_subtitles(filename):
        start, end = re.findall("([0-9]*:[0-9]*:[0-9]*,[0-9]*)", timing)
        cues.append(((convert_to_seconds(start), convert_to_seconds(end)), text))
    return cues


def split_long_cues(cues, max_words):
    """Split cues longer than ``max_words`` words into balanced chunks.

    Whole sentences wrap into two or three cramped lines on a phone, so each
    chunk gets a share of its cue's time proportional to its length. Text
    without spaces (CJK) stays one cue; ``max_words`` <= 0 keeps every cue.
    """
    if max_words <= 0:
        return cues
    result = []
    for (start, end), text in cues:
        words = text.split()
        if len(words) <= max_words:
            result.append(((start, end), text))
            continue
        count = -(-len(words) // max_words)
        size, extra = divmod(len(words), count)
        chunks, position = [], 0
        for index in range(count):
            take = size + (1 if index < extra else 0)
            chunks.append(" ".join(words[position : position + take]))
            position += take
        total = sum(len(chunk) for chunk in chunks)
        chunk_start = start
        for index, chunk in enumerate(chunks):
            chunk_end = (
                end
                if index == count - 1
                else chunk_start + (end - start) * len(chunk) / total
            )
            result.append(((chunk_start, chunk_end), chunk))
            chunk_start = chunk_end
    return result
