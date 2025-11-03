#!/usr/bin/env python3
"""Investigate why references don't match."""

import json
import re
from pathlib import Path

# Load files
with open('answers.json', 'r') as f:
    actual = json.load(f)

with open('qa_output/sample_answers.json', 'r') as f:
    expected = json.load(f)

with open('qa_output/sample_questions.json', 'r') as f:
    questions = json.load(f)

print("=" * 80)
print("DETAILED REFERENCE INVESTIGATION")
print("=" * 80)

# Build expected citation map
expected_citation_map = {}
for ans in expected:
    refs = re.findall(r'(\d+)\. ([^\n]+)', ans.get('answer', ''))
    for num, filename in refs:
        filename = filename.strip().rstrip('.')
        if "FAKE" not in filename.upper():
            if filename not in expected_citation_map:
                expected_citation_map[filename] = int(num)

print(f"\nExpected Citation Map ({len(expected_citation_map)} files):")
for fname, num in sorted(expected_citation_map.items(), key=lambda x: x[1])[:15]:
    print(f"  [{num}] {fname}")

# Build actual citation map
actual_citation_map = {}
for ans in actual[:10]:
    refs = re.findall(r'(\d+)\. ([^\n]+)', ans.get('answer', ''))
    for num, filename in refs:
        filename = filename.strip().rstrip('.')
        if "FAKE" not in filename.upper():
            if filename not in actual_citation_map:
                actual_citation_map[filename] = int(num)

print(f"\nActual Citation Map ({len(actual_citation_map)} files):")
for fname, num in sorted(actual_citation_map.items(), key=lambda x: x[1])[:15]:
    print(f"  [{num}] {fname}")

print("\n" + "=" * 80)
print("QUESTION-BY-QUESTION COMPARISON")
print("=" * 80)

for i in range(min(10, len(questions), len(expected), len(actual))):
    q = questions[i].get('question', '')
    exp = expected[i].get('answer', '')
    act = actual[i].get('answer', '')
    
    exp_refs = [(n, f.strip()) for n, f in re.findall(r'(\d+)\. ([^\n]+)', exp)]
    act_refs = [(n, f.strip()) for n, f in re.findall(r'(\d+)\. ([^\n]+)', act)]
    
    exp_files = [f.rstrip('.') for _, f in exp_refs if "FAKE" not in f.upper()]
    act_files = [f.rstrip('.') for _, f in act_refs if "FAKE" not in f.upper()]
    
    print(f"\nQ{i+1}: {q[:70]}...")
    print(f"  Expected file: {exp_files[0] if exp_files else 'none'}")
    print(f"  Actual file:   {act_files[0] if act_files else 'none'}")
    print(f"  Expected cite: {exp_refs[0][0] if exp_refs else 'none'}")
    print(f"  Actual cite:   {act_refs[0][0] if act_refs else 'none'}")
    
    if exp_files and act_files:
        if exp_files[0] != act_files[0]:
            print(f"  ⚠ FILE MISMATCH - Query found different file!")
        else:
            print(f"  ✓ File matches")
            if exp_refs[0][0] != act_refs[0][0]:
                print(f"  ⚠ CITATION MISMATCH - Same file but different citation number!")

