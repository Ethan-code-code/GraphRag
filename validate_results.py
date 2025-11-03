#!/usr/bin/env python3
"""
Validation script to compare query results with expected answers.
"""

import json
from pathlib import Path

def clean_text(text):
    """Clean text for comparison."""
    import re
    # Remove markdown formatting
    text = re.sub(r'[#*_]', '', text)
    # Remove citations
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'References.*', '', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def validate_results():
    """Validate query results against expected answers."""
    questions_file = Path("qa_output/sample_questions.json")
    expected_answers_file = Path("qa_output/sample_answers.json")
    actual_answers_file = Path("answers.json")
    
    # Load data
    with open(questions_file, 'r') as f:
        questions = json.load(f)
    
    with open(expected_answers_file, 'r') as f:
        expected_answers = json.load(f)
    
    with open(actual_answers_file, 'r') as f:
        actual_answers = json.load(f)
    
    print("=" * 80)
    print("VALIDATION REPORT")
    print("=" * 80)
    
    num_to_check = min(10, len(questions), len(expected_answers), len(actual_answers))
    
    matches = 0
    fake_detected = 0
    not_found_correct = 0
    issues = []
    
    for i in range(num_to_check):
        question = questions[i].get("question", "")
        expected = expected_answers[i].get("answer", "")
        actual = actual_answers[i].get("answer", "")
        
        # Check if it's a fake question
        is_fake = "[FAKE QUESTION" in expected or "fake_question_source" in expected
        
        # Check if expected says "not found"
        expected_not_found = "cannot be found" in expected.lower()
        actual_not_found = "cannot be found" in actual.lower()
        
        # Clean for comparison
        expected_clean = clean_text(expected)
        actual_clean = clean_text(actual)
        
        print(f"\nQ{i+1}: {question[:80]}...")
        
        if is_fake:
            if "[FAKE QUESTION" in actual or "cannot be found" in actual.lower():
                fake_detected += 1
                print("  ✓ Fake question detected correctly")
            else:
                issues.append(f"Q{i+1}: Fake question but got real answer")
                print("  ✗ Fake question should return 'not found'")
        elif expected_not_found:
            if actual_not_found:
                not_found_correct += 1
                print("  ✓ Correctly returned 'not found'")
            else:
                issues.append(f"Q{i+1}: Should be 'not found' but got answer")
                print("  ✗ Should be 'not found'")
        else:
            # Check if answers are similar (at least 30% overlap)
            if len(actual_clean) > 0 and len(expected_clean) > 0:
                # Simple similarity check
                expected_words = set(expected_clean.split())
                actual_words = set(actual_clean.split())
                if len(expected_words) > 0:
                    overlap = len(expected_words & actual_words) / len(expected_words)
                    if overlap > 0.3:
                        matches += 1
                        print(f"  ✓ Answer matches (overlap: {overlap:.1%})")
                    else:
                        issues.append(f"Q{i+1}: Low overlap ({overlap:.1%})")
                        print(f"  ⚠ Low similarity (overlap: {overlap:.1%})")
                        print(f"    Expected: {expected[:100]}...")
                        print(f"    Actual: {actual[:100]}...")
            else:
                issues.append(f"Q{i+1}: Empty answer")
                print("  ✗ Empty answer")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total questions checked: {num_to_check}")
    print(f"✓ Matching answers: {matches}/{num_to_check}")
    print(f"✓ Fake questions detected: {fake_detected}")
    print(f"✓ Correct 'not found': {not_found_correct}")
    print(f"⚠ Issues found: {len(issues)}")
    
    if issues:
        print("\nIssues:")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"  - {issue}")
    
    # Check reference formatting
    print("\n" + "=" * 80)
    print("REFERENCE FORMAT CHECK")
    print("=" * 80)
    
    ref_issues = 0
    for i, answer in enumerate(actual_answers[:num_to_check]):
        answer_text = answer.get("answer", "")
        
        # Check for duplicate references
        ref_count = answer_text.count("References")
        if ref_count > 1:
            ref_issues += 1
            print(f"  ✗ Q{i+1}: Multiple 'References' found")
        
        # Check for inline References:
        if "References:" in answer_text and "References\n" not in answer_text:
            ref_issues += 1
            print(f"  ✗ Q{i+1}: Inline 'References:' found (should be 'References\\n')")
    
    if ref_issues == 0:
        print("  ✓ All references formatted correctly")
    else:
        print(f"  ⚠ {ref_issues} reference formatting issues found")

if __name__ == "__main__":
    validate_results()

