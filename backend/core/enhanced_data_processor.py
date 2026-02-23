import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import re
from datetime import datetime
import numpy as np
from dataclasses import dataclass
from enum import Enum
import hashlib
import asyncio
from concurrent.futures import ThreadPoolExecutor
import chardet

logger = logging.getLogger(__name__)

class DatasetFormat(Enum):
    """Supported dataset formats."""
    INSTRUCTION_FOLLOWING = "instruction_following"
    CONVERSATION = "conversation"
    QUESTION_ANSWERING = "question_answering"
    TEXT_COMPLETION = "text_completion"
    CUSTOM = "custom"

class DataQualityLevel(Enum):
    """Data quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class ValidationRule:
    """Data validation rule."""
    field: str
    rule_type: str  # 'required', 'type', 'length', 'pattern', 'custom'
    condition: Any
    error_message: str
    severity: str = "error"  # 'error', 'warning'

@dataclass
class DataQualityMetrics:
    """Data quality metrics."""
    completeness_score: float
    uniqueness_score: float
    consistency_score: float
    validity_score: float
    overall_score: float
    quality_level: DataQualityLevel
    issues_found: List[Dict[str, Any]]

class EnhancedDataProcessor:
    """
    Enhanced data processor dengan multi-format support dan advanced validation.
    """
    
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.supported_formats = ["json", "jsonl", "csv", "txt", "parquet", "xlsx"]
        self.validation_rules = self._initialize_validation_rules()
        self.format_handlers = self._initialize_format_handlers()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def _initialize_validation_rules(self) -> Dict[str, List[ValidationRule]]:
        """Initialize validation rules untuk berbagai format."""
        return {
            DatasetFormat.INSTRUCTION_FOLLOWING.value: [
                ValidationRule("instruction", "required", True, "Instruction field is required"),
                ValidationRule("instruction", "type", str, "Instruction must be a string"),
                ValidationRule("instruction", "length", (1, 10000), "Instruction length must be between 1 and 10000 characters"),
                ValidationRule("output", "required", True, "Output field is required"),
                ValidationRule("output", "type", str, "Output must be a string"),
                ValidationRule("output", "length", (1, 20000), "Output length must be between 1 and 20000 characters"),
            ],
            DatasetFormat.CONVERSATION.value: [
                ValidationRule("messages", "required", True, "Messages field is required"),
                ValidationRule("messages", "type", list, "Messages must be a list"),
                ValidationRule("messages", "length", (1, 50), "Conversation must have between 1 and 50 messages"),
            ],
            DatasetFormat.QUESTION_ANSWERING.value: [
                ValidationRule("question", "required", True, "Question field is required"),
                ValidationRule("answer", "required", True, "Answer field is required"),
                ValidationRule("question", "length", (5, 2000), "Question length must be between 5 and 2000 characters"),
                ValidationRule("answer", "length", (1, 5000), "Answer length must be between 1 and 5000 characters"),
            ]
        }
    
    def _initialize_format_handlers(self) -> Dict[str, callable]:
        """Initialize format handlers."""
        return {
            "json": self._handle_json_format,
            "jsonl": self._handle_jsonl_format,
            "csv": self._handle_csv_format,
            "txt": self._handle_txt_format,
            "parquet": self._handle_parquet_format,
            "xlsx": self._handle_xlsx_format,
        }
    
    async def process_dataset_async(self, file_content: bytes, file_type: str, 
                                  format_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Process dataset secara asynchronous.
        
        Args:
            file_content: File content as bytes
            file_type: File type (json, jsonl, csv, txt, parquet, xlsx)
            format_hint: Hint untuk dataset format
            
        Returns:
            Processed dataset information
        """
        try:
            # Detect encoding
            encoding = self._detect_encoding(file_content)
            
            # Convert to string if text format
            if file_type in ["json", "jsonl", "csv", "txt"]:
                file_content_str = file_content.decode(encoding)
            else:
                file_content_str = file_content
            
            # Detect dataset format
            detected_format = self._detect_dataset_format(file_content_str, file_type, format_hint)
            
            # Process based on file type
            if file_type not in self.format_handlers:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Run processing in thread pool untuk I/O intensive operations
            loop = asyncio.get_event_loop()
            processed_data = await loop.run_in_executor(
                self.executor,
                self.format_handlers[file_type],
                file_content_str,
                detected_format
            )
            
            # Validate dataset
            validation_result = await self._validate_dataset_async(processed_data, detected_format)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(processed_data, validation_result)
            
            # Format dataset
            formatted_data = self._format_dataset(processed_data, detected_format)
            
            # Create HuggingFace Dataset
            dataset = Dataset.from_list(formatted_data)
            
            # Analyze dataset
            analysis_result = await self._analyze_dataset_async(dataset)
            
            return {
                "dataset": dataset,
                "validation": validation_result,
                "quality_metrics": quality_metrics,
                "format": detected_format,
                "analysis": analysis_result,
                "original_count": len(processed_data),
                "processed_count": len(formatted_data),
                "processing_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            raise ValueError(f"Error processing dataset: {str(e)}")
    
    def _detect_encoding(self, file_content: bytes) -> str:
        """Detect file encoding."""
        try:
            result = chardet.detect(file_content)
            return result.get('encoding', 'utf-8') or 'utf-8'
        except:
            return 'utf-8'
    
    def _detect_dataset_format(self, content: str, file_type: str, format_hint: Optional[str]) -> str:
        """Detect dataset format dari content."""
        if format_hint:
            return format_hint
        
        # Try to detect format dari content structure
        try:
            if file_type == "json" or file_type == "jsonl":
                data = json.loads(content) if file_type == "json" else json.loads(content.split('\n')[0])
                
                # Check for instruction-following format
                if isinstance(data, dict) and "instruction" in data and "output" in data:
                    return DatasetFormat.INSTRUCTION_FOLLOWING.value
                elif isinstance(data, dict) and "messages" in data:
                    return DatasetFormat.CONVERSATION.value
                elif isinstance(data, dict) and "question" in data and "answer" in data:
                    return DatasetFormat.QUESTION_ANSWERING.value
                elif isinstance(data, dict) and "text" in data:
                    return DatasetFormat.TEXT_COMPLETION.value
            
            # Default to instruction-following
            return DatasetFormat.INSTRUCTION_FOLLOWING.value
            
        except:
            return DatasetFormat.INSTRUCTION_FOLLOWING.value
    
    def _handle_json_format(self, content: str, format_type: str) -> List[Dict[str, Any]]:
        """Handle JSON format."""
        try:
            data = json.loads(content)
            
            # Handle different JSON structures
            if isinstance(data, dict):
                # Single object - convert to list
                data = [data]
            elif not isinstance(data, list):
                raise ValueError("JSON must contain a list of objects or a single object")
            
            return data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
    
    def _handle_jsonl_format(self, content: str, format_type: str) -> List[Dict[str, Any]]:
        """Handle JSONL format."""
        try:
            lines = content.strip().split('\n')
            data = []
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON on line {i+1}: {str(e)}")
            
            return data
            
        except Exception as e:
            raise ValueError(f"Error processing JSONL dataset: {str(e)}")
    
    def _handle_csv_format(self, content: str, format_type: str) -> List[Dict[str, Any]]:
        """Handle CSV format dengan advanced column mapping."""
        try:
            from io import StringIO
            df = pd.read_csv(StringIO(content))
            
            if df.empty:
                raise ValueError("CSV file is empty")
            
            # Auto-detect column mapping
            column_mapping = self._detect_csv_columns(df.columns.tolist(), format_type)
            
            # Convert to instruction format
            data = []
            for _, row in df.iterrows():
                item = self._convert_csv_row_to_format(row, column_mapping, format_type)
                data.append(item)
            
            return data
            
        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty or invalid")
        except Exception as e:
            raise ValueError(f"Error processing CSV dataset: {str(e)}")
    
    def _handle_txt_format(self, content: str, format_type: str) -> List[Dict[str, Any]]:
        """Handle plain text format."""
        try:
            lines = content.strip().split('\n')
            data = []
            
            # Try to detect text format (conversations, Q&A, etc.)
            if format_type == DatasetFormat.CONVERSATION.value:
                # Assume conversations are separated by blank lines
                conversation = []
                for line in lines:
                    line = line.strip()
                    if line:
                        # Try to detect speaker and message
                        if ":" in line:
                            speaker, message = line.split(":", 1)
                            conversation.append({
                                "role": speaker.strip().lower(),
                                "content": message.strip()
                            })
                        else:
                            conversation.append({
                                "role": "user",
                                "content": line
                            })
                    elif conversation:
                        data.append({"messages": conversation})
                        conversation = []
                
                if conversation:
                    data.append({"messages": conversation})
            
            elif format_type == DatasetFormat.QUESTION_ANSWERING.value:
                # Assume Q&A pairs are alternating lines
                for i in range(0, len(lines) - 1, 2):
                    question = lines[i].strip()
                    answer = lines[i + 1].strip() if i + 1 < len(lines) else ""
                    data.append({
                        "question": question,
                        "answer": answer
                    })
            
            else:
                # Default to instruction-following format
                # Assume each line is a training example
                for line in lines:
                    line = line.strip()
                    if line:
                        data.append({
                            "instruction": "Complete the following text",
                            "output": line
                        })
            
            return data
            
        except Exception as e:
            raise ValueError(f"Error processing TXT dataset: {str(e)}")
    
    def _handle_parquet_format(self, content: bytes, format_type: str) -> List[Dict[str, Any]]:
        """Handle Parquet format."""
        try:
            from io import BytesIO
            df = pd.read_parquet(BytesIO(content))
            
            if df.empty:
                raise ValueError("Parquet file is empty")
            
            # Convert to list of dictionaries
            data = df.to_dict('records')
            return data
            
        except Exception as e:
            raise ValueError(f"Error processing Parquet dataset: {str(e)}")
    
    def _handle_xlsx_format(self, content: bytes, format_type: str) -> List[Dict[str, Any]]:
        """Handle Excel format."""
        try:
            from io import BytesIO
            df = pd.read_excel(BytesIO(content))
            
            if df.empty:
                raise ValueError("Excel file is empty")
            
            # Convert to list of dictionaries
            data = df.to_dict('records')
            return data
            
        except Exception as e:
            raise ValueError(f"Error processing Excel dataset: {str(e)}")
    
    def _detect_csv_columns(self, columns: List[str], format_type: str) -> Dict[str, str]:
        """Auto-detect CSV column mapping."""
        mapping = {}
        
        if format_type == DatasetFormat.INSTRUCTION_FOLLOWING.value:
            # Instruction-following format
            instruction_candidates = ["instruction", "prompt", "question", "query", "task", "input_text"]
            output_candidates = ["output", "response", "answer", "completion", "target", "output_text"]
            input_candidates = ["input", "context", "background", "info", "system"]
            
        elif format_type == DatasetFormat.CONVERSATION.value:
            # Conversation format
            instruction_candidates = ["messages", "conversation", "dialogue", "chat"]
            output_candidates = ["response", "reply", "answer"]
            
        elif format_type == DatasetFormat.QUESTION_ANSWERING.value:
            # Q&A format
            instruction_candidates = ["question", "query", "problem"]
            output_candidates = ["answer", "solution", "response"]
            input_candidates = ["context", "passage", "background"]
            
        else:
            # Default to instruction-following
            instruction_candidates = ["instruction", "prompt", "question", "query", "task"]
            output_candidates = ["output", "response", "answer", "completion", "target"]
            input_candidates = ["input", "context", "background", "info"]
        
        # Find best matches
        for candidate in instruction_candidates:
            matches = [col for col in columns if candidate.lower() in col.lower()]
            if matches:
                mapping["instruction"] = matches[0]
                break
        
        for candidate in output_candidates:
            matches = [col for col in columns if candidate.lower() in col.lower()]
            if matches:
                mapping["output"] = matches[0]
                break
        
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
    
    def _convert_csv_row_to_format(self, row: pd.Series, column_mapping: Dict[str, str], 
                                  format_type: str) -> Dict[str, Any]:
        """Convert CSV row to target format."""
        item = {}
        
        if format_type == DatasetFormat.INSTRUCTION_FOLLOWING.value:
            item["instruction"] = str(row.get(column_mapping["instruction"], "")).strip()
            item["output"] = str(row.get(column_mapping["output"], "")).strip()
            if "input" in column_mapping:
                item["input"] = str(row.get(column_mapping["input"], "")).strip()
        
        elif format_type == DatasetFormat.QUESTION_ANSWERING.value:
            item["question"] = str(row.get(column_mapping["instruction"], "")).strip()
            item["answer"] = str(row.get(column_mapping["output"], "")).strip()
            if "input" in column_mapping:
                item["context"] = str(row.get(column_mapping["input"], "")).strip()
        
        else:
            # Default to instruction-following
            item["instruction"] = str(row.get(column_mapping["instruction"], "")).strip()
            item["output"] = str(row.get(column_mapping["output"], "")).strip()
            if "input" in column_mapping:
                item["input"] = str(row.get(column_mapping["input"], "")).strip()
        
        return item
    
    async def _validate_dataset_async(self, data: List[Dict[str, Any]], format_type: str) -> Dict[str, Any]:
        """Validate dataset secara asynchronous."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._validate_dataset,
            data,
            format_type
        )
    
    def _validate_dataset(self, data: List[Dict[str, Any]], format_type: str) -> Dict[str, Any]:
        """Validate dataset dengan comprehensive rules."""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {
                "total_items": len(data),
                "valid_items": 0,
                "invalid_items": 0,
                "empty_fields": {},
                "field_lengths": {},
                "data_types": {}
            }
        }
        
        if not data:
            validation_result["valid"] = False
            validation_result["errors"].append("Dataset is empty")
            return validation_result
        
        # Get validation rules for format
        rules = self.validation_rules.get(format_type, [])
        
        # Track field statistics
        field_stats = self._calculate_field_statistics(data)
        validation_result["statistics"].update(field_stats)
        
        # Validate each item
        valid_items = 0
        for i, item in enumerate(data):
            item_valid = True
            
            # Apply validation rules
            for rule in rules:
                try:
                    rule_valid = self._apply_validation_rule(item, rule, i)
                    if not rule_valid and rule.severity == "error":
                        item_valid = False
                except Exception as e:
                    validation_result["errors"].append(f"Item {i}: Validation error for rule {rule.field}: {str(e)}")
                    if rule.severity == "error":
                        item_valid = False
            
            if item_valid:
                valid_items += 1
            else:
                validation_result["statistics"]["invalid_items"] += 1
        
        validation_result["statistics"]["valid_items"] = valid_items
        
        # Overall validation check
        if validation_result["statistics"]["invalid_items"] > len(data) * 0.1:  # More than 10% invalid
            validation_result["valid"] = False
            validation_result["errors"].append(f"Too many invalid items: {validation_result['statistics']['invalid_items']}/{len(data)}")
        
        return validation_result
    
    def _calculate_field_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate field statistics."""
        stats = {
            "empty_fields": {},
            "field_lengths": {},
            "data_types": {}
        }
        
        if not data:
            return stats
        
        # Get all unique fields
        all_fields = set()
        for item in data:
            all_fields.update(item.keys())
        
        # Calculate statistics for each field
        for field in all_fields:
            values = [item.get(field) for item in data if field in item]
            
            # Empty field count
            empty_count = sum(1 for v in values if v is None or (isinstance(v, str) and not v.strip()))
            stats["empty_fields"][field] = empty_count
            
            # Field lengths (for text fields)
            if values and isinstance(values[0], str):
                lengths = [len(v) for v in values if isinstance(v, str)]
                if lengths:
                    stats["field_lengths"][field] = {
                        "min": min(lengths),
                        "max": max(lengths),
                        "mean": sum(lengths) / len(lengths),
                        "median": sorted(lengths)[len(lengths) // 2]
                    }
            
            # Data types
            types = set(type(v).__name__ for v in values if v is not None)
            stats["data_types"][field] = list(types)
        
        return stats
    
    def _apply_validation_rule(self, item: Dict[str, Any], rule: ValidationRule, 
                               item_index: int) -> bool:
        """Apply validation rule to item."""
        field_value = item.get(rule.field)
        
        if rule.rule_type == "required":
            if field_value is None or (isinstance(field_value, str) and not field_value.strip()):
                return False
        
        elif rule.rule_type == "type":
            if field_value is not None and not isinstance(field_value, rule.condition):
                return False
        
        elif rule.rule_type == "length":
            if field_value is not None and isinstance(field_value, str):
                min_len, max_len = rule.condition
                if len(field_value) < min_len or len(field_value) > max_len:
                    return False
        
        elif rule.rule_type == "pattern":
            if field_value is not None and isinstance(field_value, str):
                if not re.match(rule.condition, field_value):
                    return False
        
        elif rule.rule_type == "custom":
            # Custom validation function
            if not rule.condition(field_value, item):
                return False
        
        return True
    
    def _calculate_quality_metrics(self, data: List[Dict[str, Any]], 
                                 validation_result: Dict[str, Any]) -> DataQualityMetrics:
        """Calculate data quality metrics."""
        if not data:
            return DataQualityMetrics(
                completeness_score=0.0,
                uniqueness_score=0.0,
                consistency_score=0.0,
                validity_score=0.0,
                overall_score=0.0,
                quality_level=DataQualityLevel.CRITICAL,
                issues_found=[]
            )
        
        # Completeness score
        total_fields = sum(len(item) for item in data)
        empty_fields = sum(validation_result["statistics"]["empty_fields"].values())
        completeness_score = max(0.0, (total_fields - empty_fields) / total_fields) if total_fields > 0 else 0.0
        
        # Validity score
        valid_items = validation_result["statistics"]["valid_items"]
        validity_score = valid_items / len(data) if len(data) > 0 else 0.0
        
        # Consistency score (based on field presence)
        all_fields = set()
        for item in data:
            all_fields.update(item.keys())
        
        field_consistency_scores = []
        for field in all_fields:
            present_count = sum(1 for item in data if field in item and item[field] is not None)
            field_consistency = present_count / len(data)
            field_consistency_scores.append(field_consistency)
        
        consistency_score = sum(field_consistency_scores) / len(field_consistency_scores) if field_consistency_scores else 0.0
        
        # Uniqueness score (simplified - based on instruction uniqueness)
        if len(data) > 0 and "instruction" in data[0]:
            instructions = [item.get("instruction", "") for item in data]
            unique_instructions = len(set(instructions))
            uniqueness_score = unique_instructions / len(instructions) if instructions else 0.0
        else:
            uniqueness_score = 1.0  # Assume unique if no instruction field
        
        # Overall score
        overall_score = (completeness_score * 0.3 + 
                        validity_score * 0.4 + 
                        consistency_score * 0.2 + 
                        uniqueness_score * 0.1)
        
        # Quality level
        if overall_score >= 0.9:
            quality_level = DataQualityLevel.EXCELLENT
        elif overall_score >= 0.8:
            quality_level = DataQualityLevel.GOOD
        elif overall_score >= 0.6:
            quality_level = DataQualityLevel.FAIR
        elif overall_score >= 0.4:
            quality_level = DataQualityLevel.POOR
        else:
            quality_level = DataQualityLevel.CRITICAL
        
        # Collect issues
        issues_found = []
        for error in validation_result["errors"]:
            issues_found.append({"type": "error", "message": error})
        for warning in validation_result["warnings"]:
            issues_found.append({"type": "warning", "message": warning})
        
        return DataQualityMetrics(
            completeness_score=completeness_score,
            uniqueness_score=uniqueness_score,
            consistency_score=consistency_score,
            validity_score=validity_score,
            overall_score=overall_score,
            quality_level=quality_level,
            issues_found=issues_found
        )
    
    def _format_dataset(self, data: List[Dict[str, Any]], format_type: str) -> List[Dict[str, str]]:
        """Format dataset ke instruction-following format."""
        formatted_data = []
        
        for item in data:
            formatted_item = self._format_item(item, format_type)
            if formatted_item:
                formatted_data.append(formatted_item)
        
        return formatted_data
    
    def _format_item(self, item: Dict[str, Any], format_type: str) -> Optional[Dict[str, str]]:
        """Format single item."""
        if format_type == DatasetFormat.INSTRUCTION_FOLLOWING.value:
            return self._format_instruction_item(item)
        elif format_type == DatasetFormat.CONVERSATION.value:
            return self._format_conversation_item(item)
        elif format_type == DatasetFormat.QUESTION_ANSWERING.value:
            return self._format_qa_item(item)
        elif format_type == DatasetFormat.TEXT_COMPLETION.value:
            return self._format_text_completion_item(item)
        else:
            # Default to instruction-following
            return self._format_instruction_item(item)
    
    def _format_instruction_item(self, item: Dict[str, Any]) -> Dict[str, str]:
        """Format item ke instruction-following format."""
        instruction = str(item.get("instruction", "")).strip()
        output = str(item.get("output", "")).strip()
        input_text = str(item.get("input", "")).strip() if item.get("input") else ""
        
        # Skip if essential fields are missing
        if not instruction or not output:
            return None
        
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
    
    def _format_conversation_item(self, item: Dict[str, Any]) -> Dict[str, str]:
        """Format item ke conversation format."""
        messages = item.get("messages", [])
        if not messages or not isinstance(messages, list):
            return None
        
        # Convert conversation to instruction-following format
        conversation_text = ""
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            conversation_text += f"### {role.capitalize()}:\n{content}\n\n"
        
        # Use last message as output if it's an assistant message
        if messages and messages[-1].get("role") == "assistant":
            # Find last user message
            last_user_idx = -1
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    last_user_idx = i
                    break
            
            if last_user_idx >= 0:
                instruction = messages[last_user_idx].get("content", "")
                output = messages[-1].get("content", "")
                
                # Build conversation history as input
                input_text = ""
                for i in range(last_user_idx):
                    role = messages[i].get("role", "user")
                    content = messages[i].get("content", "")
                    input_text += f"### {role.capitalize()}:\n{content}\n\n"
                
                return self._format_instruction_item({
                    "instruction": instruction,
                    "output": output,
                    "input": input_text
                })
        
        # If can't format properly, skip
        return None
    
    def _format_qa_item(self, item: Dict[str, Any]) -> Dict[str, str]:
        """Format item ke question-answering format."""
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        context = str(item.get("context", "")).strip() if item.get("context") else ""
        
        if not question or not answer:
            return None
        
        return self._format_instruction_item({
            "instruction": question,
            "output": answer,
            "input": context
        })
    
    def _format_text_completion_item(self, item: Dict[str, Any]) -> Dict[str, str]:
        """Format item ke text completion format."""
        text = str(item.get("text", "")).strip()
        
        if not text:
            return None
        
        # Split text into prompt and completion (simple approach)
        # Use first 50% as prompt, rest as completion
        split_point = len(text) // 2
        prompt = text[:split_point]
        completion = text[split_point:]
        
        return self._format_instruction_item({
            "instruction": "Complete the following text",
            "output": completion,
            "input": prompt
        })
    
    async def tokenize_dataset_async(self, dataset: Dataset, max_length: int = 512) -> Dataset:
        """Tokenize dataset secara asynchronous."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._tokenize_dataset,
            dataset,
            max_length
        )
    
    def _tokenize_dataset(self, dataset: Dataset, max_length: int = 512) -> Dataset:
        """Tokenize dataset untuk training."""
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
    
    def create_train_val_split(self, dataset: Dataset, validation_split: float = 0.1, 
                              stratify_by: Optional[str] = None) -> DatasetDict:
        """
        Create train-validation split dengan optional stratification.
        
        Args:
            dataset: Full dataset
            validation_split: Validation split ratio
            stratify_by: Field to stratify by (if available)
            
        Returns:
            DatasetDict dengan train dan validation splits
        """
        # Shuffle dataset
        dataset = dataset.shuffle(seed=42)
        
        if stratify_by and stratify_by in dataset[0]:
            # Stratified split (simplified)
            # For now, use regular split
            split_dataset = dataset.train_test_split(
                test_size=validation_split,
                shuffle=True,
                seed=42
            )
        else:
            # Regular split
            split_dataset = dataset.train_test_split(
                test_size=validation_split,
                shuffle=True,
                seed=42
            )
        
        return DatasetDict({
            "train": split_dataset["train"],
            "validation": split_dataset["test"]
        })
    
    async def _analyze_dataset_async(self, dataset: Dataset) -> Dict[str, Any]:
        """Analyze dataset secara asynchronous."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._analyze_dataset,
            dataset
        )
    
    def _analyze_dataset(self, dataset: Dataset) -> Dict[str, Any]:
        """Analyze dataset characteristics."""
        # Calculate text lengths
        text_lengths = [len(item["text"]) for item in dataset]
        prompt_lengths = [len(item["prompt"]) for item in dataset]
        completion_lengths = [len(item["completion"]) for item in dataset]
        
        # Calculate token lengths if available
        token_lengths = []
        if "input_ids" in dataset[0]:
            token_lengths = [len(item["input_ids"]) for item in dataset]
        
        # Calculate statistics
        analysis = {
            "total_items": len(dataset),
            "text_length": {
                "mean": sum(text_lengths) / len(text_lengths),
                "min": min(text_lengths),
                "max": max(text_lengths),
                "total": sum(text_lengths),
                "std": np.std(text_lengths) if len(text_lengths) > 1 else 0
            },
            "prompt_length": {
                "mean": sum(prompt_lengths) / len(prompt_lengths),
                "min": min(prompt_lengths),
                "max": max(prompt_lengths),
                "std": np.std(prompt_lengths) if len(prompt_lengths) > 1 else 0
            },
            "completion_length": {
                "mean": sum(completion_lengths) / len(completion_lengths),
                "min": min(completion_lengths),
                "max": max(completion_lengths),
                "std": np.std(completion_lengths) if len(completion_lengths) > 1 else 0
            }
        }
        
        if token_lengths:
            analysis["token_length"] = {
                "mean": sum(token_lengths) / len(token_lengths),
                "min": min(token_lengths),
                "max": max(token_lengths),
                "std": np.std(token_lengths) if len(token_lengths) > 1 else 0
            }
        
        # Add recommendations
        analysis["recommendations"] = self._generate_analysis_recommendations(analysis)
        analysis["analysis_timestamp"] = datetime.now().isoformat()
        
        return analysis
    
    def _generate_analysis_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Text length recommendations
        avg_text_length = analysis["text_length"]["mean"]
        if avg_text_length < 50:
            recommendations.append("Consider using longer training examples for better model performance")
        elif avg_text_length > 5000:
            recommendations.append("Consider using shorter training examples to improve training efficiency")
        
        # Completion length recommendations
        avg_completion_length = analysis["completion_length"]["mean"]
        if avg_completion_length < 10:
            recommendations.append("Consider using longer completions for better model learning")
        elif avg_completion_length > 1000:
            recommendations.append("Consider using shorter completions to improve training stability")
        
        # Dataset size recommendations
        total_items = analysis["total_items"]
        if total_items < 100:
            recommendations.append("Consider using more training examples (at least 100) for better results")
        elif total_items > 10000:
            recommendations.append("Large dataset detected - consider using data sampling or curriculum learning")
        
        # Token length recommendations
        if "token_length" in analysis:
            avg_token_length = analysis["token_length"]["mean"]
            if avg_token_length > 512:
                recommendations.append("Consider increasing max_length parameter or truncating longer sequences")
        
        return recommendations
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported formats."""
        return self.supported_formats.copy()
    
    def get_format_info(self, format_type: str) -> Dict[str, Any]:
        """Get information about specific format."""
        format_info = {
            DatasetFormat.INSTRUCTION_FOLLOWING.value: {
                "description": "Instruction-following format with instruction and output",
                "required_fields": ["instruction", "output"],
                "optional_fields": ["input", "system"],
                "example": {
                    "instruction": "Translate the following text to French",
                    "input": "Hello, how are you?",
                    "output": "Bonjour, comment allez-vous?"
                }
            },
            DatasetFormat.CONVERSATION.value: {
                "description": "Conversation format with messages",
                "required_fields": ["messages"],
                "optional_fields": [],
                "example": {
                    "messages": [
                        {"role": "user", "content": "Hello!"},
                        {"role": "assistant", "content": "Hi there! How can I help you?"}
                    ]
                }
            },
            DatasetFormat.QUESTION_ANSWERING.value: {
                "description": "Question-answering format",
                "required_fields": ["question", "answer"],
                "optional_fields": ["context"],
                "example": {
                    "question": "What is the capital of France?",
                    "context": "France is a country in Europe.",
                    "answer": "Paris"
                }
            }
        }
        
        return format_info.get(format_type, {})
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)