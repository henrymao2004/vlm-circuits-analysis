"""
Script to generate Chain-of-Thought (CoT) data for object counting task.

This script:
1. Loads existing counting data from CSV
2. Generates CoT reasoning using the VLM
3. Analyzes and labels token types (visual/connector/reasoning) using DeepSeek API
4. Extracts key positions for further analysis
5. Saves the CoT prompts with metadata
"""

import csv
import os
import sys
import re
import torch
import argparse
import json
import time
from tqdm import tqdm
from typing import List, Dict, Optional
from PIL import Image
from openai import OpenAI

sys.path.append("./third_party/TransformerLens")
import transformer_lens as lens

from vision_language_prompts import VLPrompt
from general_utils import (
    get_content_key_for_prompt_dict,
    load_image_for_model,
    get_tokens,
    set_deterministic,
)
from analysis_utils import load_model
from object_counting_utils import OBJECT_TYPES


# Token classification categories
VISUAL_KEYWORDS = set([
    "see", "image", "picture", "visible", "shown", "depicted", "appears",
    "observe", "count", "identify", "spot", "notice", "viewing"
] + OBJECT_TYPES)

CONNECTOR_KEYWORDS = set([
    "therefore", "thus", "so", "hence", "consequently", "accordingly",
    "then", "next", "after", "finally", "in", "total", "altogether",
    "and", "plus", "+", "="
])

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}

# Initialize DeepSeek API client
DEEPSEEK_API_KEY = "sk-26e8562f70e746baa2c9a6af276bfd3a"
deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)


def create_cot_prompt(object_type: str, content_key: str) -> dict:
    """
    Create a CoT prompt for counting objects.
    
    Args:
        object_type: The type of object to count (e.g., "apple", "dog")
        content_key: The key for the content field ("text" or "content")
    
    Returns:
        A prompt dictionary for apply_chat_template
    """
    base_question = f'How many "{object_type}" are in the image?'
    cot_instruction = "Let's count step by step. First, carefully look at the image and identify each object one by one. Then provide the total count."
    
    prompt_dict = {
        "role": "user",
        content_key: base_question + " " + cot_instruction
    }
    
    return prompt_dict


def classify_tokens_with_deepseek(
    tokens: List[str], 
    cot_response: str,
    object_type: str,
    max_retries: int = 3
) -> List[str]:
    """
    Classify tokens using DeepSeek API for more intelligent categorization.
    
    Args:
        tokens: List of token strings
        cot_response: The full CoT reasoning text
        object_type: The object being counted
        max_retries: Maximum number of API retry attempts
    
    Returns:
        List of token types corresponding to each token
    """
    # Create a prompt for DeepSeek to classify tokens
    prompt = f"""You are analyzing a Chain-of-Thought reasoning process for counting objects in an image.

Task: The model is counting "{object_type}" objects in an image.

CoT Response: "{cot_response}"

Tokens (numbered): {json.dumps([f"{i}: {tok}" for i, tok in enumerate(tokens)], indent=2)}

Please classify each token into ONE of these categories:
1. "visual" - Tokens related to visual perception, seeing, or identifying objects in the image
2. "connector" - Logical connectors, transition words (therefore, thus, so, and, plus, total, etc.)
3. "reasoning" - Reasoning process tokens (counting steps, intermediate calculations)
4. "answer" - The final answer tokens (numbers at the end)

Return ONLY a JSON array of classifications, one per token, in the same order. Example format:
["reasoning", "visual", "visual", "connector", "reasoning", "answer"]

Your response (JSON array only):"""

    for attempt in range(max_retries):
        try:
            response = deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing reasoning chains. Respond only with valid JSON arrays."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                stream=False
            )
            
            # Parse the response
            content = response.choices[0].message.content.strip()
            
            # Extract JSON array from response (handle markdown code blocks)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            # Remove any leading/trailing text, keep only the JSON array
            content = content.strip()
            if not content.startswith('['):
                # Try to find the JSON array in the content
                match = re.search(r'\[.*\]', content, re.DOTALL)
                if match:
                    content = match.group(0)
            
            token_types = json.loads(content)
            
            # Validate length
            if len(token_types) != len(tokens):
                print(f"Warning: Token count mismatch. Expected {len(tokens)}, got {len(token_types)}. Retrying...")
                time.sleep(1)
                continue
            
            # Validate all types are valid
            valid_types = {"visual", "connector", "reasoning", "answer"}
            if all(t in valid_types for t in token_types):
                return token_types
            else:
                print(f"Warning: Invalid token types found. Retrying...")
                time.sleep(1)
                continue
                
        except Exception as e:
            print(f"Error calling DeepSeek API (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print("Falling back to rule-based classification...")
                return classify_tokens_fallback(tokens)
    
    # If all retries failed, use fallback
    print("All API attempts failed, using fallback classification")
    return classify_tokens_fallback(tokens)


def classify_tokens_fallback(tokens: List[str]) -> List[str]:
    """
    Fallback rule-based token classification.
    
    Args:
        tokens: List of token strings
    
    Returns:
        List of token types
    """
    token_types = []
    total_tokens = len(tokens)
    
    for position, token in enumerate(tokens):
        token_lower = token.lower().strip()
        
        # Check if it's in the last 10 tokens (likely answer)
        if position >= total_tokens - 10:
            if token_lower in NUMBER_WORDS or token.isdigit():
                token_types.append("answer")
                continue
        
        # Check for visual keywords
        is_visual = False
        for keyword in VISUAL_KEYWORDS:
            if keyword in token_lower:
                token_types.append("visual")
                is_visual = True
                break
        if is_visual:
            continue
        
        # Check for connector keywords
        if token_lower in CONNECTOR_KEYWORDS:
            token_types.append("connector")
            continue
        
        # Check if it's a number in reasoning
        if token.isdigit() or token_lower in NUMBER_WORDS:
            token_types.append("reasoning")
            continue
        
        # Default to reasoning
        token_types.append("reasoning")
    
    return token_types


def extract_key_positions(
    tokens: List[str],
    token_types: List[str]
) -> Dict[str, List[int]]:
    """
    Extract key token positions for different categories.
    
    Args:
        tokens: List of token strings
        token_types: List of token type labels
    
    Returns:
        Dictionary mapping category names to position indices
    """
    key_positions = {
        "visual": [],
        "connector": [],
        "reasoning": [],
        "answer": []
    }
    
    for idx, (token, token_type) in enumerate(zip(tokens, token_types)):
        if token_type in key_positions:
            key_positions[token_type].append(idx)
    
    # Find the last occurrence of numbers (likely the final answer)
    for idx in range(len(tokens) - 1, -1, -1):
        token = tokens[idx].strip().lower()
        if token.isdigit() or token in NUMBER_WORDS:
            key_positions["key_answer"] = [idx]
            break
    
    return key_positions


def generate_cot_response(
    model: lens.HookedVLTransformer,
    processor,
    prompt: str,
    image: Image.Image,
    max_new_tokens: int = 200,
    temperature: float = 0.1
) -> str:
    """
    Generate CoT reasoning response from the model.
    
    Args:
        model: The VLM model
        processor: The processor
        prompt: The prompt string
        image: The input image
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Generated text response
    """
    # Process inputs
    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True
    ).to(model.cfg.device)
    
    # Generate with the model
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    
    # Decode the response (skip the input prompt)
    input_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[0, input_length:]
    response = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response.strip()


def extract_final_answer(cot_response: str) -> Optional[str]:
    """
    Extract the final numerical answer from CoT response.
    
    Args:
        cot_response: The generated CoT reasoning text
    
    Returns:
        The extracted answer as string, or None if not found
    """
    # Try to find numbers at the end
    # Look for patterns like "Total: 5" or "total is seven" or just "7"
    patterns = [
        r'[Tt]otal[:\s]+(\w+)',
        r'[Aa]nswer[:\s]+(\w+)',
        r'[Tt]here (?:are|is)\s+(\w+)',
        r'\b(\d+)\s*\.?\s*$',  # digit at the end
        r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten)\s*\.?\s*$'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, cot_response)
        if match:
            answer = match.group(1).lower().strip()
            # Convert number words to digits
            if answer in NUMBER_WORDS:
                return str(NUMBER_WORDS[answer])
            elif answer.isdigit():
                return answer
    
    # Last resort: find any number in the last sentence
    sentences = cot_response.split('.')
    if sentences:
        last_sentence = sentences[-1]
        numbers = re.findall(r'\b\d+\b', last_sentence)
        if numbers:
            return numbers[-1]
        
        # Check for number words
        for word in last_sentence.lower().split():
            if word in NUMBER_WORDS:
                return str(NUMBER_WORDS[word])
    
    return None


def process_counting_data_with_cot(
    model: lens.HookedVLTransformer,
    processor,
    data_csv_path: str,
    images_dir: str,
    output_csv_path: str,
    max_samples: Optional[int] = None,
    image_size: tuple = (252, 252)
):
    """
    Process counting data and generate CoT responses.
    
    Args:
        model: The VLM model
        processor: The processor
        data_csv_path: Path to input CSV
        images_dir: Directory containing images
        output_csv_path: Path to output CSV with CoT data
        max_samples: Maximum number of samples to process (None for all)
        image_size: Target image size
    """
    print(f"Loading data from {data_csv_path}")
    
    # Read the original data
    data_rows = []
    with open(data_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_rows.append(row)
    
    if max_samples:
        data_rows = data_rows[:max_samples]
    
    print(f"Processing {len(data_rows)} samples...")
    
    # Get content key for this model
    content_key = get_content_key_for_prompt_dict(model.cfg.model_name)
    data_csv_dir = os.path.dirname(data_csv_path)
    
    cot_data = []
    
    for row in tqdm(data_rows, desc="Generating CoT"):
        relative_image_path = row['relative_image_path']
        original_prompt = row['prompt']
        gt_answer = row['gt_answer']
        
        # Extract object type from prompt
        match = re.search(r'How many ["\'](\w+)["\']', original_prompt)
        if not match:
            print(f"Could not extract object type from: {original_prompt}")
            continue
        
        object_type = match.group(1)
        
        # Load image
        image_path = os.path.join(data_csv_dir, relative_image_path)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        image = load_image_for_model(
            image_path,
            model.model_name,
            target_size=image_size
        )
        
        # Create CoT prompt
        cot_prompt_dict = create_cot_prompt(object_type, content_key)
        cot_prompt = processor.apply_chat_template(
            [cot_prompt_dict],
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Generate CoT response
        try:
            cot_response = generate_cot_response(
                model, processor, cot_prompt, image,
                max_new_tokens=200, temperature=0.1
            )
        except Exception as e:
            print(f"Error generating CoT for {image_path}: {e}")
            continue
        
        # Extract final answer from CoT
        pred_answer = extract_final_answer(cot_response)
        
        # Tokenize the CoT response for analysis (not the full prompt)
        tokens = get_tokens(processor, cot_response)
        
        # Classify token types using DeepSeek API
        print(f"  Classifying tokens for {object_type} (sample {len(cot_data)+1})...")
        token_types = classify_tokens_with_deepseek(
            tokens=tokens,
            cot_response=cot_response,
            object_type=object_type
        )
        
        # Extract key positions
        key_positions = extract_key_positions(tokens, token_types)
        
        # Create metadata
        metadata = {
            'tokens': tokens,
            'token_types': token_types,
            'key_positions': key_positions,
            'cot_response': cot_response,
            'original_prompt': original_prompt,
            'object_type': object_type
        }
        
        cot_data.append({
            'relative_image_path': relative_image_path,
            'prompt': cot_prompt,
            'cot_response': cot_response,
            'gt_answer': gt_answer,
            'pred_answer': pred_answer if pred_answer else "N/A",
            'object_type': object_type,
            'tokens': '|'.join(tokens),
            'token_types': '|'.join(token_types),
            'visual_positions': ','.join(map(str, key_positions['visual'])),
            'connector_positions': ','.join(map(str, key_positions['connector'])),
            'answer_positions': ','.join(map(str, key_positions['answer'])),
            'key_answer_position': str(key_positions.get('key_answer', [])[-1]) if key_positions.get('key_answer') else ""
        })
    
    # Write to output CSV
    print(f"Writing {len(cot_data)} CoT samples to {output_csv_path}")
    
    fieldnames = [
        'relative_image_path', 'prompt', 'cot_response', 'gt_answer', 
        'pred_answer', 'object_type', 'tokens', 'token_types',
        'visual_positions', 'connector_positions', 'answer_positions',
        'key_answer_position'
    ]
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cot_data)
    
    print(f"CoT data generation complete! Saved to {output_csv_path}")
    
    # Print some statistics
    correct_preds = sum(1 for row in cot_data if row['pred_answer'] == row['gt_answer'])
    print(f"\nStatistics:")
    print(f"  Total samples: {len(cot_data)}")
    print(f"  Correct predictions: {correct_preds}/{len(cot_data)} ({100*correct_preds/len(cot_data):.1f}%)")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CoT data for counting task")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="qwen2-7b-vl-instruct",
        help="Model name"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model"
    )
    parser.add_argument(
        "--data_csv",
        type=str,
        default="./data/counting/qwen2-7b-vl-instruct_visual_data.csv",
        help="Path to input counting data CSV"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="./data/counting/images",
        help="Directory containing images"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./data/counting/qwen2-7b-vl-instruct_cot_data.csv",
        help="Path to output CoT data CSV"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Maximum number of samples to process (default: 100 due to API limits)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed
    set_deterministic(args.seed)
    
    print("="*60)
    print("Chain-of-Thought Data Generation")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Model path: {args.model_path}")
    print(f"Input data: {args.data_csv}")
    print(f"Output data: {args.output_csv}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # Load model
    print("\nLoading model...")
    model, processor = load_model(
        model_name=args.model_name,
        model_path=args.model_path,
        device=args.device,
        use_tlens_wrapper=True,
        extra_hooks=True,
        torch_dtype=torch.bfloat16
    )
    model.eval()
    print("Model loaded successfully!")
    
    # Process data
    process_counting_data_with_cot(
        model=model,
        processor=processor,
        data_csv_path=args.data_csv,
        images_dir=args.images_dir,
        output_csv_path=args.output_csv,
        max_samples=args.max_samples
    )
    
    print("\n" + "="*60)
    print("CoT data generation completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

