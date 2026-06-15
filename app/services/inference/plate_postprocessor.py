"""Rule-based Vietnamese license plate OCR postprocessor.

Format: NN-YY-ZZZZZ
  NN    = 2 chữ số (mã tỉnh) - CHẮC CHẮN SỐ
  YY    = 1-2 ký tự series (chữ + số mix) - CÓ THỂ CHỮ HOẶC SỐ
  ZZZZZ = 4-5 chữ số cuối - CHẮC CHẮN SỐ
"""
import re
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
from statistics import median

from app.core.logger import get_logger

logger = get_logger(__name__)

MIN_PLATE_SCORE = 60

VALID_PROVINCE_CODES = {str(i).zfill(2) for i in range(11, 100)} - {"13"}

ALLOWED_SERIES_LETTERS = set("ABCDEFGHKLMNPSTUVXYZ")
DISALLOWED_SERIES_LETTERS = set("IJOQRW")

# Context-specific confusion maps
TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1", "T": "1",
    "Z": "2",
    "A": "4",
    "S": "5",
    "G": "6", "C": "6",
    "B": "8",
}

TO_LETTER = {
    "0": "D", "1": "T", "2": "Z",
    "4": "A", "5": "S", "6": "G", "8": "B",
}

PLATE_PATTERNS = [
    re.compile(r"^(?P<prov>\d{2})(?P<ser>[A-Z]\d)(?P<num>\d{5})$"),
    re.compile(r"^(?P<prov>\d{2})(?P<ser>[A-Z]{2})(?P<num>\d{5})$"),
    re.compile(r"^(?P<prov>\d{2})(?P<ser>[A-Z]\d{2})(?P<num>\d{5})$"),
    re.compile(r"^(?P<prov>\d{2})(?P<ser>[A-Z]\d)(?P<num>\d{4})$"),
    re.compile(r"^(?P<prov>\d{2})(?P<ser>[A-Z]{1,2}\d?)(?P<num>\d{4,5})$"),
]


@dataclass
class OCRChar:
    char: str
    conf: float
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class PlateResult:
    raw_text: str
    normalized_text: str
    score: float
    valid: bool
    corrections: List[dict]


# ---------------------------------------------------------------------------
# Character sorting
# ---------------------------------------------------------------------------
def sort_ocr_chars(chars: List[OCRChar]) -> List[OCRChar]:
    if len(chars) <= 1:
        return chars

    heights = [c.h for c in chars]
    med_h = median(heights) if heights else 20
    threshold = med_h * 0.6

    sorted_by_y = sorted(chars, key=lambda c: c.cy)
    lines: List[List[OCRChar]] = []
    current_line = [sorted_by_y[0]]

    for c in sorted_by_y[1:]:
        if abs(c.cy - current_line[0].cy) <= threshold:
            current_line.append(c)
        else:
            lines.append(current_line)
            current_line = [c]
    lines.append(current_line)

    lines.sort(key=lambda line: min(c.cy for c in line))
    result = []
    for line in lines:
        result.extend(sorted(line, key=lambda c: c.cx))
    return result


def raw_text_from_chars(chars: List[OCRChar]) -> str:
    return "".join(c.char for c in chars)


def avg_confidence(chars: List[OCRChar]) -> float:
    if not chars:
        return 0.0
    return sum(c.conf for c in chars) / len(chars)


# ---------------------------------------------------------------------------
# Position-aware correction
# ---------------------------------------------------------------------------
def _to_digit(ch: str) -> Tuple[str, Optional[dict]]:
    """Force character to digit."""
    if ch.isdigit():
        return ch, None
    fixed = TO_DIGIT.get(ch.upper())
    if fixed:
        return fixed, {"from": ch, "to": fixed, "reason": "must be digit"}
    return ch, None


def _to_series_letter(ch: str) -> Tuple[str, Optional[dict]]:
    """Fix character that MUST be a series letter (pos 2). Digits get converted."""
    upper = ch.upper()
    if upper in ALLOWED_SERIES_LETTERS:
        return upper, None

    # Digit → letter
    if ch.isdigit():
        candidate = TO_LETTER.get(ch)
        if candidate and candidate in ALLOWED_SERIES_LETTERS:
            return candidate, {"from": ch, "to": candidate, "reason": "must be series letter"}

    # Disallowed letter → allowed letter
    if upper in DISALLOWED_SERIES_LETTERS:
        fb = {"I": "T", "O": "D", "Q": "D", "J": "T", "R": "P", "W": "V"}.get(upper, upper)
        if fb in ALLOWED_SERIES_LETTERS:
            return fb, {"from": ch, "to": fb, "reason": "disallowed series letter"}

    candidate = TO_LETTER.get(ch)
    if candidate and candidate in ALLOWED_SERIES_LETTERS:
        return candidate, {"from": ch, "to": candidate, "reason": "series letter"}

    return upper, None


def _to_series_any(ch: str) -> Tuple[str, Optional[dict]]:
    """Fix character for series pos 3+ (letter or digit both OK).
    Disallowed letters get fixed or converted to digit."""
    upper = ch.upper()
    if upper in ALLOWED_SERIES_LETTERS or upper.isdigit():
        return upper, None

    # Disallowed letter → try digit first (I→1 is more common than I→T here)
    if upper in DISALLOWED_SERIES_LETTERS:
        d = TO_DIGIT.get(upper)
        if d:
            return d, {"from": ch, "to": d, "reason": "disallowed→digit"}

    candidate = TO_LETTER.get(ch)
    if candidate and candidate in ALLOWED_SERIES_LETTERS:
        return candidate, {"from": ch, "to": candidate, "reason": "series letter"}

    return upper, None


def correct_plate_text(raw: str) -> Tuple[str, List[dict]]:
    """Position-aware correction:
      pos 0-1:  MUST be digits (province code)
      pos 2:    MUST be series letter
      pos 3:    series (letter or digit, keep as-is if valid)
      pos 4+:   MUST be digits (number tail)
    
    For plates with single-char series (e.g. 30H-12345), series_end=3.
    For plates with 2-char series (e.g. 29B1-12345), series_end=4.
    """
    clean = raw.upper().replace("-", "").replace(" ", "").replace(".", "")
    if len(clean) < 5:
        return raw, []

    chars = list(clean)
    n = len(chars)
    corrections = []

    # -- Province (pos 0,1): MUST be digits --
    for i in range(min(2, n)):
        fixed, corr = _to_digit(chars[i])
        if corr:
            corr["pos"] = i
            corrections.append(corr)
        chars[i] = fixed

    # -- First series char (pos 2): MUST be letter --
    if n > 2:
        fixed, corr = _to_series_letter(chars[2])
        if corr:
            corr["pos"] = 2
            corrections.append(corr)
        chars[2] = fixed

    # -- Determine series_end: where do tail digits begin? --
    series_end = 3
    if n > 3:
        c3 = chars[3]
        if c3.isalpha() and c3.upper() in ALLOWED_SERIES_LETTERS:
            series_end = 4
        elif c3.isdigit():
            # Could be series digit (B1) or tail digit
            if n >= 9:
                series_end = 4
            elif n == 8:
                series_end = 4
            else:
                series_end = 3
        else:
            # Disallowed or ambiguous → fix
            fixed, corr = _to_series_any(c3)
            chars[3] = fixed
            if corr:
                corr["pos"] = 3
                corrections.append(corr)
            if fixed.isalpha():
                series_end = 4
            else:
                series_end = 4

    # -- Tail (pos series_end..n): MUST be digits, no exceptions --
    for i in range(series_end, n):
        fixed, corr = _to_digit(chars[i])
        if corr:
            corr["pos"] = i
            corrections.append(corr)
        chars[i] = fixed

    return "".join(chars), corrections


# ---------------------------------------------------------------------------
# Validation + scoring
# ---------------------------------------------------------------------------
def _match_pattern(text: str) -> Optional[dict]:
    clean = text.replace("-", "").replace(" ", "")
    for pat in PLATE_PATTERNS:
        m = pat.match(clean)
        if m:
            return m.groupdict()
    return None


def score_plate(corrected: str, corrections: List[dict], avg_conf: float) -> Tuple[float, bool]:
    score = 0.0
    clean = corrected.replace("-", "").replace(" ", "")

    # Length
    if 7 <= len(clean) <= 10:
        score += 10
    else:
        score -= 5

    # Province code
    if len(clean) >= 2:
        prov = clean[:2]
        if prov.isdigit() and prov in VALID_PROVINCE_CODES:
            score += 30
        elif prov.isdigit():
            score -= 10  # valid digits but unknown province
        else:
            score -= 25

    # Pattern match
    parts = _match_pattern(corrected)
    if parts:
        score += 5

        num = parts.get("num", "")
        if num.isdigit():
            score += 20
        else:
            score -= 20  # tail MUST be digits

        ser = parts.get("ser", "")
        if ser:
            valid_ser = all(c in ALLOWED_SERIES_LETTERS or c.isdigit() for c in ser)
            if valid_ser:
                score += 15
            elif any(c in DISALLOWED_SERIES_LETTERS for c in ser if c.isalpha()):
                score -= 15
    else:
        if len(clean) >= 7:
            tail = clean[-5:] if len(clean) >= 9 else clean[-4:]
            if tail.isdigit():
                score += 10
            else:
                score -= 20

    # Confidence
    if avg_conf > 0.7:
        score += 10
    elif avg_conf > 0.5:
        score += 5

    # Correction penalty
    score -= len(corrections) * 3

    return score, score >= MIN_PLATE_SCORE


def normalize_plate(corrected: str) -> str:
    clean = corrected.replace("-", "").replace(" ", "")

    parts = _match_pattern(clean)
    if parts:
        return f"{parts['prov']}-{parts['ser']}-{parts['num']}"

    # Fallback manual split
    if len(clean) >= 7:
        prov = clean[:2]
        # Find series end
        se = 2
        for i in range(2, min(5, len(clean))):
            if clean[i].isalpha():
                se = i + 1
            elif clean[i].isdigit() and i > 2:
                if i == 3 and len(clean) >= 9:
                    se = i + 1
                else:
                    break
            else:
                break

        ser = clean[2:se]
        num = clean[se:]
        if num:
            return f"{prov}-{ser}-{num}"

    return corrected


# ---------------------------------------------------------------------------
# Image enhancement
# ---------------------------------------------------------------------------
def enhance_plate_crop(crop: np.ndarray) -> List[np.ndarray]:
    variants = [crop]
    h, w = crop.shape[:2]

    up = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    variants.append(up)

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    variants.append(cv2.filter2D(up, -1, kernel))

    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY) if len(up.shape) == 3 else up
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    variants.append(cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR))

    return variants


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def postprocess_plate(raw_detections: List[dict]) -> PlateResult:
    if not raw_detections:
        return PlateResult("", "", 0, False, [])

    chars = []
    for d in raw_detections:
        x1, y1, x2, y2 = d["bbox"]
        chars.append(OCRChar(
            char=d["char"], conf=d["conf"],
            cx=(x1 + x2) / 2, cy=(y1 + y2) / 2,
            w=x2 - x1, h=y2 - y1,
        ))

    sorted_chars = sort_ocr_chars(chars)
    raw_text = raw_text_from_chars(sorted_chars)
    conf = avg_confidence(sorted_chars)

    corrected, corrections = correct_plate_text(raw_text)
    score, valid = score_plate(corrected, corrections, conf)
    normalized = normalize_plate(corrected) if valid else corrected

    return PlateResult(raw_text, normalized, score, valid, corrections)
