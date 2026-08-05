"""After Effects keyframe data: the interchange half of todo 18.

CSV and JSON already existed. What did not was a format any compositor will
read, so these tests are written against the *format's* rules — the header line
AE looks for, tab separation, CRLF, and the gap handling that stops a missing
frame importing as a jump to the top-left corner.
"""

from __future__ import annotations

import math

import pytest

from stable.core.trackdata import after_effects_keyframes


def _document(**overrides) -> str:
    kwargs = {"fps": 24.0, "width": 1920, "height": 1080}
    kwargs.update(overrides)
    tracks = kwargs.pop("tracks", [("p1", [(10.0, 20.0), (11.0, 21.0)])])
    return after_effects_keyframes(tracks, **kwargs)


def test_the_document_opens_and_closes_the_way_after_effects_expects():
    text = _document()
    lines = text.split("\r\n")
    assert lines[0] == "Adobe After Effects 8.0 Keyframe Data"
    assert "End of Keyframe Data" in lines


def test_it_is_tab_separated_with_crlf_endings():
    """AE rejects LF-only keyframe data pasted on Windows."""
    text = _document()
    assert "\r\n" in text
    assert "\n" not in text.replace("\r\n", "")
    assert "\tFrame\tX pixels\tY pixels\tZ pixels" in text


def test_the_source_size_and_rate_are_declared():
    text = _document(fps=29.97, width=1280, height=720)
    assert "\tUnits Per Second\t29.97" in text
    assert "\tSource Width\t1280" in text
    assert "\tSource Height\t720" in text


def test_positions_are_written_frame_indexed_and_unflipped():
    """AE measures from the top-left with Y down, exactly as this project does."""
    text = _document(tracks=[("p1", [(10.0, 20.0), (11.5, 21.25)])])
    assert "\t0\t10.0000\t20.0000\t0" in text
    assert "\t1\t11.5000\t21.2500\t0" in text


def test_a_gap_is_skipped_rather_than_written_as_zero():
    nan = math.nan
    text = _document(tracks=[("p1", [(10.0, 20.0), (nan, nan), (12.0, 22.0)])])
    frames = [line.split("\t")[1] for line in text.split("\r\n") if line.startswith("\t0\t")]
    assert frames == ["0"]  # frame 0 present
    assert "\t1\t" not in text  # the gap is absent, not (0, 0)
    assert "\t2\t12.0000\t22.0000\t0" in text


def test_every_point_gets_its_own_block():
    text = _document(tracks=[("p1", [(1.0, 2.0)]), ("p2", [(3.0, 4.0)])])
    assert text.count("Transform\tPosition") == 2
    assert "Transform\tPosition\tp1" in text
    assert "Transform\tPosition\tp2" in text


def test_no_tracks_still_produces_a_well_formed_document():
    text = after_effects_keyframes([], fps=24.0, width=640, height=480)
    assert text.startswith("Adobe After Effects 8.0 Keyframe Data")
    assert "End of Keyframe Data" in text


def test_a_zero_frame_rate_is_refused():
    """Units Per Second is what makes the frame numbers mean anything."""
    with pytest.raises(ValueError, match="fps"):
        _document(fps=0.0)
