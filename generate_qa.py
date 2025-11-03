"""
Question-Answer Generation Script
Reads files from a folder and uses LLM (Ollama) to generate Q&A pairs.
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import time

try:
    import random
    import tqdm
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install: pip install tqdm")
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_N_REAL_QUESTIONS = 300  # Real questions from files
DEFAULT_N_FAKE_QUESTIONS = 100  # Fake questions not in files
DEFAULT_N_QUESTIONS = 400  # Total questions


def extract_text_from_file(file_path: Path) -> str:
    """
    Extract text content from various file types.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Extracted text content
    """
    try:
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Extract text from JSON (handle various structures)
                if isinstance(data, dict):
                    # Extract all string values
                    text_parts = [str(v) for v in data.values() if isinstance(v, str)]
                    return ' '.join(text_parts)
                elif isinstance(data, list):
                    text_parts = [str(item) for item in data if isinstance(item, str)]
                    return ' '.join(text_parts)
                else:
                    return str(data)
        
        elif file_path.suffix.lower() in ['.md', '.txt', '.markdown']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        elif file_path.suffix.lower() == '.pdf':
            try:
                import PyPDF2
                text = ""
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                return text
            except ImportError:
                logger.warning("PyPDF2 not installed. Skipping PDF file. Install with: pip install PyPDF2")
                return ""
        
        else:
            # Try to read as text
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except:
                logger.warning(f"Could not read file as text: {file_path}")
                return ""
    
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return ""


def collect_all_files(input_folder: Path) -> List[Path]:
    """
    Recursively collect all files from input folder.
    
    Args:
        input_folder: Path to input folder
        
    Returns:
        List of file paths
    """
    files = []
    
    if not input_folder.exists():
        logger.error(f"Input folder does not exist: {input_folder}")
        return files
    
    # Supported file extensions
    supported_extensions = ['.md', '.txt', '.json', '.markdown', '.pdf']
    
    for root, dirs, filenames in os.walk(input_folder):
        for filename in filenames:
            file_path = Path(root) / filename
            if file_path.suffix.lower() in supported_extensions or file_path.suffix == '':
                files.append(file_path)
    
    return files


def is_valid_sentence(sentence: str) -> bool:
    """
    Validate if a sentence is suitable for question generation.
    
    Args:
        sentence: Sentence to validate
        
    Returns:
        True if sentence is valid, False otherwise
    """
    sentence = sentence.strip()
    
    # Too short or too long
    if len(sentence) < 20 or len(sentence) > 300:
        return False
    
    # Contains HTML/XML tags
    if re.search(r'<[^>]+>', sentence):
        return False
    
    # Contains HTML entities
    if re.search(r'&[a-zA-Z]+;', sentence):
        return False
    
    # Contains metadata markers
    invalid_patterns = [
        r'</?page_(start|end)>',
        r'</?footer>',
        r'</?header>',
        r'^###\s+',
        r'^##\s+',
        r'^#\s+',
        r'^\d+$',  # Just a number
        r'^[<>]+$',  # Just brackets
        r'^[^a-zA-Z]*$',  # No letters
    ]
    
    for pattern in invalid_patterns:
        if re.search(pattern, sentence, re.IGNORECASE):
            return False
    
    # Check if it's meaningful (has enough words)
    words = sentence.split()
    if len(words) < 5:  # At least 5 words
        return False
    
    # Check if it starts with meaningful words (not just special chars)
    first_words = ' '.join(words[:3]).lower()
    if re.match(r'^[^a-zA-Z\s]+', first_words):
        return False
    
    return True


def clean_sentence(sentence: str) -> str:
    """
    Clean sentence by removing HTML/XML tags and normalizing.
    
    Args:
        sentence: Raw sentence
        
    Returns:
        Cleaned sentence
    """
    # Remove HTML/XML tags
    sentence = re.sub(r'<[^>]+>', '', sentence)
    # Remove HTML entities
    sentence = re.sub(r'&[a-zA-Z]+;', '', sentence)
    # Normalize whitespace
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    return sentence


def sentence_to_question(sentence: str) -> str:
    """
    Convert a sentence to a proper question format.
    
    Args:
        sentence: Input sentence
        
    Returns:
        Question string
    """
    sentence = clean_sentence(sentence)
    sentence = sentence.strip()
    
    # Remove trailing punctuation
    sentence = sentence.rstrip('.!?')
    
    if not sentence:
        return ""
    
    # "The maximum duration is 15 years" -> "What is the maximum duration?"
    if re.search(r'\bis\b', sentence, re.IGNORECASE):
        # Pattern: "The X is Y" -> "What is X?"
        match = re.search(r'^(the|a|an)\s+([^.]+?)\s+is\s+(.+)$', sentence, re.IGNORECASE)
        if match:
            subject = match.group(2).strip()
            # Clean subject
            subject = re.sub(r'^(the|a|an)\s+', '', subject, flags=re.IGNORECASE)
            return f"What is {subject}?"
        
        # Pattern: "X is Y" -> "What is X?"
        match = re.search(r'^([A-Z][^.]+?)\s+is\s+(.+)$', sentence)
        if match and len(match.group(1).split()) <= 8:
            subject = match.group(1).strip()
            return f"What is {subject}?"
    
    # "The filing fee for appeal is 200 USD" -> "What is the filing fee for appeal?"
    if 'for' in sentence.lower() and 'is' in sentence.lower():
        match = re.search(r'the\s+([^for]+?)\s+for\s+([^is]+?)\s+is\s+(.+)$', sentence, re.IGNORECASE)
        if match:
            return f"What is the {match.group(1).strip()} for {match.group(2).strip()}?"
    
    # "X requires Y" -> "What does X require?"
    if 'requires' in sentence.lower():
        parts = sentence.split('requires', 1)
        if len(parts) == 2 and len(parts[0].strip().split()) <= 8:
            subject = parts[0].strip()
            return f"What does {subject} require?"
    
    # "X must Y" -> "What must X do?" or "What is required of X?"
    if ' must ' in sentence.lower():
        parts = sentence.split(' must ', 1)
        if len(parts) == 2 and len(parts[0].strip().split()) <= 8:
            subject = parts[0].strip()
            return f"What must {subject} do?"
    
    # "X shall Y" -> "What shall X do?"
    if ' shall ' in sentence.lower():
        parts = sentence.split(' shall ', 1)
        if len(parts) == 2 and len(parts[0].strip().split()) <= 8:
            subject = parts[0].strip()
            return f"What shall {subject} do?"
    
    # "X can be Y" -> "What can X be?"
    if ' can be ' in sentence.lower():
        parts = sentence.split(' can be ', 1)
        if len(parts) == 2 and len(parts[0].strip().split()) <= 8:
            subject = parts[0].strip()
            return f"What can {subject} be?"
    
    # "According to X, Y" -> "What does X state?"
    if sentence.lower().startswith('according to'):
        match = re.search(r'according\s+to\s+([^,]+)', sentence, re.IGNORECASE)
        if match:
            return f"What does {match.group(1).strip()} state?"
    
    # Extract key phrase for question (first 10-12 words)
    words = sentence.split()
    if len(words) > 10:
        key_phrase = ' '.join(words[:10])
    else:
        key_phrase = sentence
    
    # Remove articles from start
    key_phrase = re.sub(r'^(the|a|an)\s+', '', key_phrase, flags=re.IGNORECASE)
    
    return f"What is {key_phrase}?"


def extract_random_sentences_from_files(files_data: List[Dict], n: int = 300) -> List[Dict]:
    """
    Extract random sentences from random paragraphs in random files.
    
    Args:
        files_data: List of dicts with 'text', 'file_name', 'file_path'
        n: Number of sentences to extract
        
    Returns:
        List of dicts with 'sentence', 'file_name', 'paragraph_index'
    """
    sentences = []
    
    # Shuffle files for randomness
    random.shuffle(files_data)
    
    logger.info(f"Extracting {n} random sentences from files...")
    
    attempt = 0
    max_attempts = n * 3  # Prevent infinite loop
    
    while len(sentences) < n and attempt < max_attempts:
        attempt += 1
        
        # Pick random file
        if not files_data:
            break
        
        file_data = random.choice(files_data)
        text = file_data.get('text', '')
        file_name = file_data.get('file_name', 'unknown')
        
        if len(text.strip()) < 50:
            continue
        
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 20]
        
        if not paragraphs:
            continue
        
        # Pick random paragraph
        paragraph = random.choice(paragraphs)
        
        # Clean paragraph of HTML/XML tags first
        paragraph_clean = re.sub(r'<[^>]+>', '', paragraph)
        paragraph_clean = re.sub(r'&[a-zA-Z]+;', '', paragraph_clean)
        paragraph_clean = re.sub(r'\s+', ' ', paragraph_clean).strip()
        
        if len(paragraph_clean) < 30:
            continue
        
        # Split paragraph into sentences
        sentence_endings = re.compile(r'[.!?]+\s+')
        para_sentences = [s.strip() for s in sentence_endings.split(paragraph_clean) if len(s.strip()) > 20]
        
        if not para_sentences:
            continue
        
        # Filter valid sentences only
        valid_sentences = [s for s in para_sentences if is_valid_sentence(s)]
        
        if not valid_sentences:
            continue
        
        # Pick random sentence from valid ones
        sentence = random.choice(valid_sentences)
        
        # Clean the sentence
        sentence = clean_sentence(sentence)
        
        # Skip if already collected
        if any(s['sentence'] == sentence for s in sentences):
            continue
        
        # Generate a proper question
        question = sentence_to_question(sentence)
        
        # Validate question quality
        if not question or len(question) < 10 or len(question) > 200:
            continue
        
        # Skip questions that are too generic or unclear
        if question.lower().startswith('what is what is') or '###' in question or '<' in question:
            continue
        
        sentences.append({
            'sentence': sentence,
            'question': question,
            'file_name': file_name,
            'paragraph': paragraph_clean[:200],  # Store preview
            'answer': sentence  # Answer is the sentence itself
        })
    
    logger.info(f"Extracted {len(sentences)} random sentences")
    return sentences


def generate_fake_questions(n: int = 100) -> List[Dict]:
    """
    Generate fake questions that don't exist in the files.
    
    Args:
        n: Number of fake questions to generate
        
    Returns:
        List of fake question-answer pairs
    """
    fake_questions = [
        "What is the capital of Mars?",
        "How many moons does Jupiter have?",
        "What is the speed of light in vacuum?",
        "Who invented the telephone?",
        "What is the chemical formula for water?",
        "What is the largest ocean on Earth?",
        "How many continents are there?",
        "What is the boiling point of nitrogen?",
        "Who wrote the novel '1984'?",
        "What is the atomic number of gold?",
        "What is the formula for calculating area of a circle?",
        "Who painted the Mona Lisa?",
        "What is the distance from Earth to the Sun?",
        "What is the molecular weight of oxygen?",
        "Who discovered penicillin?",
        "What is the freezing point of mercury?",
        "How many planets are in our solar system?",
        "What is the speed of sound?",
        "Who composed the Moonlight Sonata?",
        "What is the largest mammal on Earth?",
    ]
    
    # Generate more variations
    fake_answers = [
        "There is no definitive answer to this question. [FAKE QUESTION - This question was generated for testing and does not exist in the source documents.]",
        "This information is not available in the provided documents. [FAKE QUESTION - This question was generated for testing and does not exist in the source documents.]",
        "The answer to this question cannot be found in the source material. [FAKE QUESTION - This question was generated for testing and does not exist in the source documents.]",
        "No relevant information exists regarding this topic. [FAKE QUESTION - This question was generated for testing and does not exist in the source documents.]",
        "This question cannot be answered based on the available documents. [FAKE QUESTION - This question was generated for testing and does not exist in the source documents.]",
    ]
    
    fake_qa = []
    
    # Use predefined questions
    for i, question in enumerate(fake_questions[:min(n, len(fake_questions))]):
        fake_answer = random.choice(fake_answers)
        # Ensure fake marker is present
        if "[FAKE QUESTION" not in fake_answer:
            fake_answer = f"{fake_answer} [FAKE QUESTION - This question was generated for testing and does not exist in the source documents.]"
        
        fake_qa.append({
            'question': question,
            'answer': fake_answer,
            'file_name': None,
            'is_fake': True
        })
    
    # Generate more random fake questions
    question_templates = [
        "What is the {noun} of {subject}?",
        "How many {noun} does {subject} have?",
        "Who {verb} {object}?",
        "When did {event} happen?",
        "Where is {location} located?",
        "Why does {subject} {verb}?",
        "What are the benefits of {topic}?",
        "How does {process} work?",
    ]
    
    subjects = ['government', 'company', 'organization', 'system', 'process', 'method']
    nouns = ['purpose', 'function', 'role', 'structure', 'composition', 'nature']
    verbs = ['created', 'developed', 'established', 'formed', 'initiated']
    objects = ['this system', 'the process', 'the method', 'the organization']
    events = ['this occur', 'this happen', 'this take place', 'this begin']
    locations = ['this place', 'that location', 'the site', 'the area']
    topics = ['this approach', 'this method', 'this system', 'this process']
    processes = ['this mechanism', 'this procedure', 'this system', 'this method']
    
    while len(fake_qa) < n:
        template = random.choice(question_templates)
        
        try:
            if '{noun}' in template and '{subject}' in template:
                question = template.format(noun=random.choice(nouns), subject=random.choice(subjects))
            elif '{verb}' in template and '{object}' in template:
                question = template.format(verb=random.choice(verbs), object=random.choice(objects))
            elif '{subject}' in template and '{verb}' in template:
                question = template.format(subject=random.choice(subjects), verb=random.choice(verbs))
            elif '{event}' in template:
                question = template.format(event=random.choice(events))
            elif '{location}' in template:
                question = template.format(location=random.choice(locations))
            elif '{topic}' in template:
                question = template.format(topic=random.choice(topics))
            elif '{process}' in template:
                question = template.format(process=random.choice(processes))
            elif '{subject}' in template:
                question = template.format(subject=random.choice(subjects))
            else:
                # Fallback: use template as-is or add random subject
                question = template.replace('{subject}', random.choice(subjects))
                question = question.replace('{noun}', random.choice(nouns))
                question = question.replace('{verb}', random.choice(verbs))
        except (KeyError, ValueError) as e:
            logger.debug(f"Error formatting template '{template}': {e}")
            # Skip this template and try another
            continue
        
        # Check if we already have this question
        if not any(q['question'] == question for q in fake_qa):
            # Get random fake answer
            fake_answer = random.choice(fake_answers)
            # Ensure it has the fake marker
            if "[FAKE QUESTION" not in fake_answer:
                fake_answer = f"{fake_answer} [FAKE QUESTION - This question was generated for testing and does not exist in the source documents.]"
            
            fake_qa.append({
                'question': question,
                'answer': fake_answer,
                'file_name': None,
                'is_fake': True
            })
    
    logger.info(f"Generated {len(fake_qa)} fake questions")
    return fake_qa


def generate_questions(files_data: List[Dict], n_real: int = 300, n_fake: int = 100) -> List[Dict]:
    """
    Generate questions from random sentences in files and fake questions.
    
    Args:
        files_data: List of dicts with file text and metadata
        n_real: Number of real questions from files
        n_fake: Number of fake questions
        
    Returns:
        List of question-answer dictionaries
    """
    all_qa_pairs = []
    
    # Generate real questions from random sentences
    real_sentences = extract_random_sentences_from_files(files_data, n=n_real)
    
    for sent_data in real_sentences:
        sentence = sent_data['sentence']
        file_name = sent_data['file_name']
        answer = sent_data['answer']
        
        # Get question (already generated in extract function)
        question = sent_data.get('question', sentence_to_question(sentence))
        
        # Validate question one more time
        if not question or len(question) < 10:
            continue
        
        # Determine question type based on content
        q_type = "factual"
        if any(word in question.lower() for word in ['why', 'how', 'what does', 'what can', 'what must']):
            q_type = "reasoning"
        elif any(word in question.lower() for word in ['who', 'when', 'where']):
            q_type = "contextual"
        
        # Clean answer (remove HTML/XML if any)
        clean_answer = clean_sentence(answer)
        
        all_qa_pairs.append({
            "question": question,
            "type": q_type,
            "category": random.choice(["objective", "analytical"]),
            "answer": clean_answer,
            "source_file": file_name,
            "is_fake": False
        })
    
    # Generate fake questions
    fake_qa = generate_fake_questions(n=n_fake)
    all_qa_pairs.extend(fake_qa)
    
    # Shuffle to mix real and fake questions
    random.shuffle(all_qa_pairs)
    
    logger.info(f"Generated {len([q for q in all_qa_pairs if not q.get('is_fake', False)])} real and {len([q for q in all_qa_pairs if q.get('is_fake', False)])} fake questions")
    return all_qa_pairs


def process_files(
    input_folder: Path,
    output_folder: Path,
    n_real: int = DEFAULT_N_REAL_QUESTIONS,
    n_fake: int = DEFAULT_N_FAKE_QUESTIONS,
    debug: bool = False
) -> None:
    """
    Process all files and generate Q&A pairs.
    
    Args:
        input_folder: Path to input folder
        output_folder: Path to output folder
        n_questions: Target number of questions
        model: Ollama model name
        debug: Enable debug output
    """
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Collect all files
    logger.info(f"Scanning files in {input_folder}...")
    files = collect_all_files(input_folder)
    
    if not files:
        logger.error(f"No files found in {input_folder}")
        return
    
    logger.info(f"Found {len(files)} file(s)")
    
    # Collect all text content
    all_text_parts = []
    file_metadata = {}
    
    start_time = time.time()
    
    logger.info("Reading files and extracting text...")
    for i, file_path in enumerate(files, 1):
        logger.info(f"Processing file {i}/{len(files)}: {file_path.name}")
        
        try:
            text = extract_text_from_file(file_path)
            
            if text and len(text.strip()) > 100:
                all_text_parts.append({
                    "text": text,
                    "file_name": file_path.name,
                    "file_path": str(file_path)
                })
                file_metadata[file_path.name] = file_path
            else:
                logger.warning(f"Skipping {file_path.name}: insufficient content")
        
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            continue
    
    if not all_text_parts:
        logger.error("No valid text content extracted from files")
        return
    
    # Combine all text
    combined_text = "\n\n---\n\n".join([
        f"[File: {part['file_name']}]\n{part['text']}"
        for part in all_text_parts
    ])
    
    logger.info(f"Combined text length: {len(combined_text):,} characters")
    
    # Estimate time
    estimated_time = len(combined_text) / 1000 * 2  # Rough estimate
    logger.info(f"Estimated processing time: {estimated_time/60:.1f} minutes")
    
    # Generate questions
    logger.info(f"Generating {n_real} real + {n_fake} fake = {n_real + n_fake} Q&A pairs...")
    
    all_qa_pairs = generate_questions(
        all_text_parts,
        n_real=n_real,
        n_fake=n_fake
    )
    
    if not all_qa_pairs:
        logger.error("No Q&A pairs generated")
        return
    
    # Split into questions and answers
    questions = []
    answers = []
    citation_counter = 1
    citation_map = {}
    
    for qa in all_qa_pairs:
        # Check if it's a fake question
        is_fake = qa.get('is_fake', False)
        
        # Extract source file from answer or use default
        if is_fake:
            source_file = "fake_question_source"
        else:
            source_file = qa.get("source_file") or "document"
        
        # Extract file name from answer if present (only for real questions)
        answer_text = qa.get("answer", "")
        if not is_fake:
            source_match = re.search(r'\[Source:\s*([^\]]+)\]', answer_text)
            if source_match:
                source_file = source_match.group(1).strip()
        
        # Assign citation number
        if source_file not in citation_map:
            citation_map[source_file] = citation_counter
            citation_counter += 1
        
        citation_num = citation_map[source_file]
        
        # Format question
        questions.append({
            "question": qa.get("question", ""),
            "type": qa.get("type", "factual"),
            "category": qa.get("category", "objective")
        })
        
        # Format answer with proper citation
        clean_answer = re.sub(r'\s*\[Source:[^\]]+\]', '', answer_text)
        
        if is_fake:
            # For fake questions, include the fake marker and citation
            if "[FAKE QUESTION" not in clean_answer:
                clean_answer = f"{clean_answer} [FAKE QUESTION - This question was generated for testing and does not exist in the source documents.]"
            
            if not clean_answer.strip().endswith('.'):
                clean_answer += "."
            
            formatted_answer = f"{clean_answer} [{citation_num}].\n\nReferences\n{citation_num}. {source_file} (FAKE - Question not in source documents)."
        else:
            # For real questions, normal citation
            if clean_answer and not clean_answer.strip().endswith('.'):
                clean_answer += "."
            
            formatted_answer = f"{clean_answer} [{citation_num}].\n\nReferences\n{citation_num}. {source_file}."
        
        answers.append({
            "answer": formatted_answer
        })
    
    # Save questions
    questions_file = output_folder / "sample_questions.json"
    with open(questions_file, 'w', encoding='utf-8') as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Saved {len(questions)} questions to {questions_file}")
    
    # Save answers
    answers_file = output_folder / "sample_answers.json"
    with open(answers_file, 'w', encoding='utf-8') as f:
        json.dump(answers, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Saved {len(answers)} answers to {answers_file}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"✅ Generation completed in {elapsed_time/60:.2f} minutes")
    logger.info(f"   Questions generated: {len(questions)}")
    logger.info(f"   Output folder: {output_folder}")
    logger.info(f"{'='*60}")
    
    if debug:
        logger.info("\nSample questions (first 5):")
        for i, q in enumerate(questions[:5], 1):
            logger.info(f"  {i}. [{q.get('type', 'N/A')}] {q.get('question', '')[:80]}...")
        
        logger.info("\nSample answers (first 3):")
        for i, a in enumerate(answers[:3], 1):
            answer_preview = a.get('answer', '')[:150]
            logger.info(f"  {i}. {answer_preview}...")


def main():
    """Main function with command-line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate question-answer pairs from files using Ollama LLM"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input folder path containing files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="qa_output",
        help="Output folder path (default: qa_output)"
    )
    parser.add_argument(
        "--n-real",
        type=int,
        default=DEFAULT_N_REAL_QUESTIONS,
        help=f"Number of real questions from files (default: {DEFAULT_N_REAL_QUESTIONS})"
    )
    parser.add_argument(
        "--n-fake",
        type=int,
        default=DEFAULT_N_FAKE_QUESTIONS,
        help=f"Number of fake questions (default: {DEFAULT_N_FAKE_QUESTIONS})"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to print sample questions and answers"
    )
    
    args = parser.parse_args()
    
    input_folder = Path(args.input)
    output_folder = Path(args.output)
    
    if not input_folder.exists():
        logger.error(f"Input folder does not exist: {input_folder}")
        exit(1)
    
    logger.info("="*60)
    logger.info("Question-Answer Generation")
    logger.info("="*60)
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Output folder: {output_folder}")
    logger.info(f"Real questions: {args.n_real}")
    logger.info(f"Fake questions: {args.n_fake}")
    logger.info(f"Total questions: {args.n_real + args.n_fake}")
    logger.info("="*60)
    
    try:
        process_files(
            input_folder=input_folder,
            output_folder=output_folder,
            n_real=args.n_real,
            n_fake=args.n_fake,
            debug=args.debug
        )
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.debug:
            import traceback
            logger.debug(traceback.format_exc())
        exit(1)


if __name__ == "__main__":
    main()

