"""Test plate postprocessor."""
import sys
sys.path.insert(0, "/teamspace/studios/this_studio")

from app.services.inference.plate_postprocessor import correct_plate_text, score_plate, normalize_plate

TESTS = [
    # (raw, expected_normalized, expected_valid, desc)
    ("29B112345",  "29-B1-12345", True,  "basic"),
    ("29BI1234S",  "29-B1-12345", True,  "I→1, S→5 in tail"),
    ("3OK156789",  "30-K1-56789", True,  "O→0 in province"),
    ("S9A112345",  "59-A1-12345", True,  "S→5 in province"),
    ("985188888",  "98-S1-88888", True,  "5→S in series pos"),
    ("99C2I2S4S",  "99-C2-12545", True,  "I→1 S→5 in tail"),
    ("67AB12S45",  "67-AB-12545", True,  "S→5 in tail"),
    ("13A112345",  None,          False, "province 13 invalid"),
    ("29S112345",  "29-S1-12345", True,  "normal"),
    ("30H123456",  "30-H1-23456", True,  "9-char plate"),
    ("51F112345",  "51-F1-12345", True,  "normal 9-char"),
    ("29BS12345",  "29-B5-12345", True,  "S→5 in tail (after series letter B)"),
    ("29B1S2345",  "29-B1-52345", True,  "S→5 in first tail digit"),
    ("29B1IZSO5",  "29-B1-12505", True,  "multiple fixes in tail I→1 Z→2 S→5 O→0"),
]


def main():
    print("=" * 70)
    print("PLATE POSTPROCESSOR TEST v2")
    print("  Format: NN-YY-ZZZZZ")
    print("  NN=digits, YY=mix, ZZZZZ=MUST digits")
    print("=" * 70)

    ok = fail = 0
    for raw, expected, exp_valid, desc in TESTS:
        corrected, corrections = correct_plate_text(raw)
        score, valid = score_plate(corrected, corrections, 0.8)
        normalized = normalize_plate(corrected)

        match = False
        if expected and normalized == expected:
            match = True
        elif exp_valid is not None and valid == exp_valid and expected is None:
            match = True

        status = "PASS" if match else "FAIL"
        if match:
            ok += 1
        else:
            fail += 1

        print(f"\n[{status}] {desc}")
        print(f"  raw={raw}  →  corrected={corrected}  →  norm={normalized}")
        if expected:
            print(f"  expected={expected}  {'✓' if normalized == expected else '✗'}")
        print(f"  score={score:.0f}  valid={valid}  corrections={len(corrections)}")
        for c in corrections:
            print(f"    pos{c['pos']}: {c['from']}→{c['to']} ({c['reason']})")

    print(f"\n{'=' * 70}")
    print(f"TOTAL: {ok} passed, {fail} failed / {len(TESTS)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
