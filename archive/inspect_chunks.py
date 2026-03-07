"""
inspect_chunks.py — Debugging helper for the main.py pipeline.

Runs only the first stage of main.py (regex keyword filtering) for a given
patient CSN and writes matched chunks to a human-readable text file.

No LLM calls, no AWS dependencies.

Usage:
    python3 inspect_chunks.py --csn <CSN> [--data ./patient_features.json] [--output ./chunks_<CSN>.txt]
"""

import argparse
import json
import re
import sys

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from questions_dict import question_dictionary

# Copied verbatim from main.py (lines 99-101)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=650, chunk_overlap=100, separators=["."]
)


# Copied verbatim from main.py (lines 225-256)

def filter_chunks(chunks, filters, excluders=[], filter_case=None):
    # Returns list of (chunk, first_matched_text) tuples
    filtered_notes = []
    for chunk in chunks:
        first_match_text = None
        matches = []
        for i, fil in enumerate(filters):
            flags = re.IGNORECASE
            if filter_case:
                if filter_case[i]:
                    flags = 0

            match = re.search(
                fil,
                chunk,
                flags=flags
            )
            if match:
                for exc in excluders:
                    exc_match = re.search(
                        exc,
                        chunk,
                        flags=re.IGNORECASE
                    )
                    if exc_match:
                        match = False

                if match:
                    if first_match_text is None:
                        first_match_text = match.group(0)
                    matches.append(fil)

        if matches:
            filtered_notes.append((chunk, first_match_text))

    return filtered_notes

# ---------------------------------------------------------------------------
# Data loading (mirrors main.py lines 480-519)
# ---------------------------------------------------------------------------
def load_patient_bundle(data_path: str, target_csn: str):
    """Load patient_features JSON and return the bundle for target_csn."""
    try:
        fhir_data = pd.read_json(data_path)
    except Exception as e:
        print(f"ERROR: Could not read data file '{data_path}': {e}", file=sys.stderr)
        sys.exit(1)

    bundles = [json.loads(d) for d in fhir_data['features']]

    for bundle in bundles:
        csn = bundle['demographics'][0]['csn']
        if str(csn) == str(target_csn):
            return bundle

    print(f"ERROR: CSN '{target_csn}' not found in '{data_path}'.", file=sys.stderr)
    sys.exit(1)


def extract_notes(bundle) -> list:
    """Build flat list of note strings from a patient bundle."""
    notes = []
    for note in bundle['binary']:
        if 'note_type' not in note.keys():
            txt = ''
        else:
            note_type = note['note_type']
            txt = 'Note Type: ' + note_type + '\n'
        txt += note['note']
        notes.append(txt)

    for note in bundle['medication_orders']:
        txt = 'Note Type: Medication Dosage Text \n'
        txt += note['dosage_text']
        notes.append(txt)

    return notes


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------
def build_report(csn: str, data_path: str, all_chunks: list, results: dict, show_match: bool = False) -> str:
    lines = []

    lines.append("=== CHUNK INSPECTION REPORT ===")
    lines.append(f"CSN: {csn}")
    lines.append(f"Data file: {data_path}")
    lines.append(f"Total chunks: {len(all_chunks)}")
    lines.append("")

    lines.append("SUMMARY")
    lines.append("-" * 7)
    col_width = max(len(k) for k in results) + 2
    for comp, matched in results.items():
        lines.append(f"{comp:<{col_width}}: {len(matched)} / {len(all_chunks)} chunks")
    lines.append("")

    sep = "=" * 80
    for comp, matched in results.items():
        info = question_dictionary[comp]
        filters = info.get('filters', [])
        excluders = info.get('excluders', [])

        lines.append(sep)
        lines.append(f"COMPLICATION: {comp}")
        lines.append(f"Filters  : {filters}")
        lines.append(f"Excluders: {excluders}")
        lines.append(f"Matched  : {len(matched)} / {len(all_chunks)} chunks")
        lines.append(sep)
        lines.append("")

        if matched:
            for i, (chunk, match_text) in enumerate(matched, start=1):
                lines.append(f"--- Chunk {i} ---")
                if show_match and match_text is not None:
                    lines.append(f"[MATCH: {match_text!r}]")
                lines.append(chunk)
                lines.append("")
        else:
            lines.append(f"(No matching chunks for {comp})")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Inspect regex-filtered chunks for a patient CSN (no LLM/AWS needed)."
    )
    parser.add_argument("--csn", required=True, help="Patient CSN to inspect")
    parser.add_argument("--data", default="./patient_features.json",
                        help="Path to patient_features JSON file (default: ./patient_features.json)")
    parser.add_argument("--output", default=None,
                        help="Output text file path (default: ./new_chunks_<CSN>.txt)")
    parser.add_argument("--show-match", action="store_true",
                        help="Print the matched text above each chunk in the output file")
    args = parser.parse_args()

    output_path = args.output if args.output else f"./new_chunks_{args.csn}.txt"

    print(f"Loading data for CSN {args.csn} from '{args.data}'...")
    bundle = load_patient_bundle(args.data, args.csn)

    print("Extracting notes...")
    notes = extract_notes(bundle)
    if not notes:
        print(f"ERROR: No notes found for CSN {args.csn}.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(notes)} notes. Chunking...")
    notes_str = "\n\n".join(notes)
    all_chunks = text_splitter.split_text(notes_str)
    print(f"Total chunks after splitting: {len(all_chunks)}")

    print("Applying regex filters for each complication...")
    results = {}
    for comp, info in question_dictionary.items():
        filters = info.get('filters', [])
        excluders = info.get('excluders', [])
        filter_case = info.get('filter_case')
        if filters:
            matched = filter_chunks(all_chunks, filters, excluders, filter_case=filter_case)
        else:
            matched = [(chunk, None) for chunk in all_chunks]
        results[comp] = matched

    report = build_report(args.csn, args.data, all_chunks, results, show_match=args.show_match)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nReport written to: {output_path}")
    print("\nSUMMARY:")
    col_width = max(len(k) for k in results) + 2
    for comp, matched in results.items():
        print(f"  {comp:<{col_width}}: {len(matched):>3} / {len(all_chunks)} chunks")


if __name__ == "__main__":
    main()
