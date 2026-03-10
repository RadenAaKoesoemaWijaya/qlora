"""
Async file processing utilities untuk non-blocking I/O operations.
"""

import aiofiles
import json
import csv
import io
from pathlib import Path
from typing import Dict, List, Any, Union
import logging

logger = logging.getLogger(__name__)


async def read_file_async(file_path: str, encoding: str = 'utf-8') -> str:
    """
    Read file asynchronously.
    
    Args:
        file_path: Path ke file
        encoding: File encoding
        
    Returns:
        File content sebagai string
    """
    async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
        return await f.read()


async def read_json_async(file_path: str) -> Union[Dict, List]:
    """
    Read dan parse JSON file asynchronously.
    
    Args:
        file_path: Path ke JSON file
        
    Returns:
        Parsed JSON data
    """
    content = await read_file_async(file_path)
    return json.loads(content)


async def read_jsonl_async(file_path: str) -> List[Dict]:
    """
    Read JSONL file asynchronously.
    
    Args:
        file_path: Path ke JSONL file
        
    Returns:
        List of JSON objects
    """
    content = await read_file_async(file_path)
    lines = content.strip().split('\n')
    return [json.loads(line) for line in lines if line.strip()]


async def read_csv_async(file_path: str, delimiter: str = ',') -> List[Dict[str, Any]]:
    """
    Read CSV file asynchronously.
    
    Args:
        file_path: Path ke CSV file
        delimiter: CSV delimiter
        
    Returns:
        List of dictionaries (CSV rows)
    """
    content = await read_file_async(file_path)
    
    # Parse CSV dari string
    reader = csv.DictReader(io.StringIO(content), delimiter=delimiter)
    return list(reader)


async def write_file_async(file_path: str, content: str, encoding: str = 'utf-8'):
    """
    Write file asynchronously.
    
    Args:
        file_path: Path untuk write
        content: Content untuk write
        encoding: File encoding
    """
    async with aiofiles.open(file_path, 'w', encoding=encoding) as f:
        await f.write(content)


async def write_json_async(file_path: str, data: Union[Dict, List], indent: int = 2):
    """
    Write JSON file asynchronously.
    
    Args:
        file_path: Path untuk write
        data: Data untuk serialize
        indent: JSON indentation
    """
    content = json.dumps(data, indent=indent, ensure_ascii=False)
    await write_file_async(file_path, content)


async def process_dataset_file_async(file_path: str, file_type: str) -> Dict[str, Any]:
    """
    Process dataset file based on type (JSON, JSONL, CSV).
    
    Args:
        file_path: Path ke dataset file
        file_type: Type file (JSON, JSONL, CSV, TXT, Parquet, XLSX)
        
    Returns:
        Dictionary dengan processed data dan metadata
    """
    file_type = file_type.upper()
    
    try:
        if file_type == 'JSON':
            data = await read_json_async(file_path)
            if isinstance(data, list):
                rows = len(data)
            else:
                data = [data]
                rows = 1
                
        elif file_type == 'JSONL':
            data = await read_jsonl_async(file_path)
            rows = len(data)
            
        elif file_type == 'CSV':
            data = await read_csv_async(file_path)
            rows = len(data)
            
        elif file_type == 'TXT':
            content = await read_file_async(file_path)
            # Split by double newline untuk paragraph format
            data = [{"text": para} for para in content.split('\n\n') if para.strip()]
            rows = len(data)
            
        else:
            raise ValueError(f"Unsupported file type for async processing: {file_type}")
        
        return {
            "data": data,
            "rows": rows,
            "file_type": file_type,
            "file_path": file_path,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error processing {file_type} file {file_path}: {e}")
        return {
            "data": [],
            "rows": 0,
            "file_type": file_type,
            "file_path": file_path,
            "success": False,
            "error": str(e)
        }


async def validate_dataset_async(file_path: str, file_type: str) -> Dict[str, Any]:
    """
    Validate dataset file tanpa loading seluruh content ke memory.
    
    Args:
        file_path: Path ke dataset file
        file_type: Type file
        
    Returns:
        Validation result dengan status dan issues
    """
    result = {
        "valid": False,
        "file_path": file_path,
        "file_type": file_type,
        "rows": 0,
        "issues": [],
        "sample": None
    }
    
    try:
        # Check file exists dan readable
        path = Path(file_path)
        if not path.exists():
            result["issues"].append("File does not exist")
            return result
        
        if not path.is_file():
            result["issues"].append("Path is not a file")
            return result
        
        # Check file size (max 1GB)
        file_size = path.stat().st_size
        if file_size > 1_073_741_824:  # 1GB in bytes
            result["issues"].append("File size exceeds 1GB limit")
            return result
        
        # Process file untuk validasi
        processed = await process_dataset_file_async(file_path, file_type)
        
        if not processed["success"]:
            result["issues"].append(f"Failed to parse file: {processed.get('error')}")
            return result
        
        data = processed["data"]
        
        # Check minimum rows
        if processed["rows"] < 1:
            result["issues"].append("Dataset is empty")
            return result
        
        # Check structure
        if isinstance(data, list) and len(data) > 0:
            first_item = data[0]
            if not isinstance(first_item, dict):
                result["issues"].append("Dataset items must be objects/dictionaries")
                return result
            
            # Check untuk instruction/output format
            keys = set(first_item.keys())
            has_instruction = any('instruction' in k.lower() or 'prompt' in k.lower() for k in keys)
            has_output = any('output' in k.lower() or 'response' in k.lower() or 'answer' in k.lower() for k in keys)
            
            if not (has_instruction and has_output):
                result["issues"].append("Dataset should have 'instruction'/'prompt' and 'output'/'response' fields")
        
        # Sample first item untuk preview
        if len(data) > 0:
            result["sample"] = dict(list(data[0].items())[:5])  # First 5 fields only
        
        result["valid"] = len(result["issues"]) == 0
        result["rows"] = processed["rows"]
        
    except Exception as e:
        result["issues"].append(f"Validation error: {str(e)}")
    
    return result
