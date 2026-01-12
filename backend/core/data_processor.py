import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import json
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Advanced data processor untuk QLoRA fine-tuning dengan proper validation dan formatting.
    """
    
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.supported_formats = ["json", "jsonl", "csv", "txt"]
        
    def validate_dataset_structure(self, data: List[Dict], format_type: str) -> Dict[str, Any]:
        """
        Validate dataset structure untuk instruction-following format.
        
        Args:
            data: List of data items
            format_type: Type of dataset format
            
        Returns:
            Validation result dengan statistik dan error messages
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {
                "total_items": len(data),
                "valid_items": 0,
                "invalid_items": 0,
                "average_instruction_length": 0,
                "average_output_length": 0,
                "empty_instructions": 0,
                "empty_outputs": 0
            }
        }
        
        if not data:
            validation_result["valid"] = False
            validation_result["errors"].append("Dataset is empty")
            return validation_result
        
        total_instruction_length = 0
        total_output_length = 0
        valid_items = 0
        
        for i, item in enumerate(data):
            item_valid = True
            
            # Check required fields
            if "instruction" not in item:
                validation_result["errors"].append(f"Item {i}: Missing 'instruction' field")
                item_valid = False
            
            if "output" not in item:
                validation_result["errors"].append(f"Item {i}: Missing 'output' field")
                item_valid = False
            
            # Check field types and content
            if item_valid:
                instruction = str(item.get("instruction", "")).strip()
                output = str(item.get("output", "")).strip()
                
                if not instruction:
                    validation_result["warnings"].append(f"Item {i}: Empty instruction")
                    validation_result["statistics"]["empty_instructions"] += 1
                
                if not output:
                    validation_result["warnings"].append(f"Item {i}: Empty output")
                    validation_result["statistics"]["empty_outputs"] += 1
                
                # Calculate lengths
                total_instruction_length += len(instruction)
                total_output_length += len(output)
                
                if instruction and output:
                    valid_items += 1
            
            if not item_valid:
                validation_result["statistics"]["invalid_items"] += 1
        
        # Calculate averages
        if valid_items > 0:
            validation_result["statistics"]["valid_items"] = valid_items
            validation_result["statistics"]["average_instruction_length"] = total_instruction_length / valid_items
            validation_result["statistics"]["average_output_length"] = total_output_length / valid_items
        
        # Final validation check
        if validation_result["statistics"]["invalid_items"] > len(data) * 0.1:  # More than 10% invalid
            validation_result["valid"] = False
            validation_result["errors"].append(f"Too many invalid items: {validation_result['statistics']['invalid_items']}/{len(data)}")
        
        return validation_result
    
    def process_json_dataset(self, file_content: str) -> Dict[str, Any]:
        """
        Process JSON dataset dengan proper validation dan formatting.
        
        Args:
            file_content: JSON file content as string
            
        Returns:
            Dictionary dengan dataset dan validation results
        """
        try:
            # Parse JSON
            data = json.loads(file_content)
            
            # Handle different JSON structures
            if isinstance(data, dict):
                # Single object - convert to list
                data = [data]
            elif not isinstance(data, list):
                raise ValueError("JSON must contain a list of objects or a single object")
            
            # Validate dataset
            validation_result = self.validate_dataset_structure(data, "json")
            
            if not validation_result["valid"]:
                raise ValueError(f"Dataset validation failed: {'; '.join(validation_result['errors'])}")
            
            # Format dataset
            formatted_data = []
            for item in data:
                formatted_item = self.format_instruction_item(item)
                formatted_data.append(formatted_item)
            
            # Create HuggingFace Dataset
            dataset = Dataset.from_list(formatted_data)
            
            return {
                "dataset": dataset,
                "validation": validation_result,
                "format": "instruction_following",
                "original_count": len(data),
                "processed_count": len(formatted_data)
            }
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error processing JSON dataset: {str(e)}")
    
    def process_jsonl_dataset(self, file_content: str) -> Dict[str, Any]:
        """Process JSONL dataset."""
        try:
            lines = file_content.strip().split('\n')
            data = []
            
            for i, line in enumerate(lines):
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {i+1}: {str(e)}")
            
            # Validate and format seperti JSON
            return self.process_json_dataset(json.dumps(data))
            
        except Exception as e:
            raise ValueError(f"Error processing JSONL dataset: {str(e)}")
    
    def process_csv_dataset(self, file_content: str) -> Dict[str, Any]:
        """Process CSV dataset dengan mapping kolom."""
        try:
            # Parse CSV
            from io import StringIO
            df = pd.read_csv(StringIO(file_content))
            
            if df.empty:
                raise ValueError("CSV file is empty")
            
            # Detect column mapping
            column_mapping = self.detect_column_mapping(df.columns.tolist())
            
            # Convert to instruction format
            data = []
            for _, row in df.iterrows():
                item = {
                    "instruction": str(row.get(column_mapping["instruction"], "")).strip(),
                    "output": str(row.get(column_mapping["output"], "")).strip(),
                    "input": str(row.get(column_mapping.get("input", ""), "")).strip() if column_mapping.get("input") else ""
                }
                data.append(item)
            
            # Validate and format
            return self.process_json_dataset(json.dumps(data))
            
        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty or invalid")
        except Exception as e:
            raise ValueError(f"Error processing CSV dataset: {str(e)}")
    
    def detect_column_mapping(self, columns: List[str]) -> Dict[str, str]:
        """
        Automatically detect column mapping untuk instruction-following format.
        
        Args:
            columns: List of column names
            
        Returns:
            Dictionary mapping standard names to actual column names
        """
        mapping = {}
        
        # Instruction column detection
        instruction_candidates = ["instruction", "prompt", "question", "query", "task"]
        for candidate in instruction_candidates:
            matches = [col for col in columns if candidate.lower() in col.lower()]
            if matches:
                mapping["instruction"] = matches[0]
                break
        
        # Output column detection
        output_candidates = ["output", "response", "answer", "completion", "target"]
        for candidate in output_candidates:
            matches = [col for col in columns if candidate.lower() in col.lower()]
            if matches:
                mapping["output"] = matches[0]
                break
        
        # Input column detection (optional)
        input_candidates = ["input", "context", "background", "info"]
        for candidate in input_candidates:
            matches = [col for col in columns if candidate.lower() in col.lower()]
            if matches:
                mapping["input"] = matches[0]
                break
        
        # Validate required columns
        if "instruction" not in mapping:
            raise ValueError(f"Could not detect instruction column. Available columns: {columns}")
        
        if "output" not in mapping:
            raise ValueError(f"Could not detect output column. Available columns: {columns}")
        
        return mapping
    
    def format_instruction_item(self, item: Dict[str, Any]) -> Dict[str, str]:
        """
        Format item ke instruction-following format.
        
        Args:
            item: Raw data item
            
        Returns:
            Formatted item dengan text, prompt, dan completion
        """
        instruction = str(item.get("instruction", "")).strip()
        output = str(item.get("output", "")).strip()
        input_text = str(item.get("input", "")).strip() if item.get("input") else ""
        
        # Create prompt
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        
        # Full text for training
        text = prompt + output
        
        # Add EOS token jika belum ada
        if self.tokenizer.eos_token and not text.endswith(self.tokenizer.eos_token):
            text += self.tokenizer.eos_token
        
        return {
            "text": text,
            "prompt": prompt,
            "completion": output,
            "instruction": instruction,
            "input": input_text
        }
    
    def tokenize_dataset(self, dataset: Dataset, max_length: int = 512) -> Dataset:
        """
        Tokenize dataset untuk training.
        
        Args:
            dataset: HuggingFace Dataset
            max_length: Maximum sequence length
            
        Returns:
            Tokenized dataset
        """
        def tokenize_function(examples):
            # Tokenize the text
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,  # Don't pad here, will be done by data collator
                max_length=max_length,
                return_overflowing_tokens=False,
                return_length=True,
            )
            
            # Add length information
            tokenized["length"] = [len(ids) for ids in tokenized["input_ids"]]
            
            return tokenized
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        # Filter out sequences that are too long (optional)
        if max_length:
            tokenized_dataset = tokenized_dataset.filter(
                lambda example: example["length"] <= max_length,
                desc="Filtering long sequences"
            )
        
        return tokenized_dataset
    
    def create_train_val_split(self, dataset: Dataset, validation_split: float = 0.1) -> DatasetDict:
        """
        Create train-validation split.
        
        Args:
            dataset: Full dataset
            validation_split: Validation split ratio
            
        Returns:
            DatasetDict dengan train dan validation splits
        """
        # Shuffle dataset
        dataset = dataset.shuffle(seed=42)
        
        # Split dataset
        split_dataset = dataset.train_test_split(
            test_size=validation_split,
            shuffle=True,
            seed=42
        )
        
        return DatasetDict({
            "train": split_dataset["train"],
            "validation": split_dataset["test"]
        })
    
    def analyze_dataset(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Analyze dataset characteristics.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Analysis results
        """
        # Calculate text lengths
        text_lengths = [len(item["text"]) for item in dataset]
        prompt_lengths = [len(item["prompt"]) for item in dataset]
        completion_lengths = [len(item["completion"]) for item in dataset]
        
        # Calculate token lengths if available
        token_lengths = []
        if "input_ids" in dataset[0]:
            token_lengths = [len(item["input_ids"]) for item in dataset]
        
        return {
            "total_items": len(dataset),
            "text_length": {
                "mean": sum(text_lengths) / len(text_lengths),
                "min": min(text_lengths),
                "max": max(text_lengths),
                "total": sum(text_lengths)
            },
            "prompt_length": {
                "mean": sum(prompt_lengths) / len(prompt_lengths),
                "min": min(prompt_lengths),
                "max": max(prompt_lengths)
            },
            "completion_length": {
                "mean": sum(completion_lengths) / len(completion_lengths),
                "min": min(completion_lengths),
                "max": max(completion_lengths)
            },
            "token_length": {
                "mean": sum(token_lengths) / len(token_lengths) if token_lengths else 0,
                "min": min(token_lengths) if token_lengths else 0,
                "max": max(token_lengths) if token_lengths else 0
            } if token_lengths else None,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def process_dataset_file(self, file_content: str, file_type: str) -> Dict[str, Any]:
        """
        Process dataset file berdasarkan tipe.
        
        Args:
            file_content: File content as string
            file_type: File type (json, jsonl, csv)
            
        Returns:
            Processed dataset information
        """
        file_type = file_type.lower()
        
        if file_type == "json":
            return self.process_json_dataset(file_content)
        elif file_type == "jsonl":
            return self.process_jsonl_dataset(file_content)
        elif file_type == "csv":
            return self.process_csv_dataset(file_content)
        else:
            raise ValueError(f"Unsupported file type: {file_type}. Supported types: {self.supported_formats}")