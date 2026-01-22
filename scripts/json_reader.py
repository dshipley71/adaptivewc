import json
import sys
from pathlib import Path


def print_structure(data, indent=0):
    """Recursively print the structure and contents of JSON data."""
    prefix = "  " * indent
    
    if isinstance(data, dict):
        if not data:
            print(f"{prefix}{{}}")
            return
        print(f"{prefix}{{")
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                print(f"{prefix}  \"{key}\":")
                print_structure(value, indent + 2)
            else:
                print(f"{prefix}  \"{key}\": {format_value(value)}")
        print(f"{prefix}}}")
    
    elif isinstance(data, list):
        if not data:
            print(f"{prefix}[]")
            return
        print(f"{prefix}[")
        for i, item in enumerate(data):
            if isinstance(item, (dict, list)):
                print_structure(item, indent + 1)
            else:
                print(f"{prefix}  {format_value(item)}")
            if i < len(data) - 1:
                pass  # Could add comma handling here
        print(f"{prefix}]")
    
    else:
        print(f"{prefix}{format_value(data)}")


def format_value(value):
    """Format a primitive value for display."""
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, str):
        return f"\"{value}\""
    else:
        return str(value)


def get_type_summary(data, path="root"):
    """Get a summary of types in the JSON structure."""
    summary = []
    
    if isinstance(data, dict):
        summary.append(f"{path}: object ({len(data)} keys)")
        for key, value in data.items():
            summary.extend(get_type_summary(value, f"{path}.{key}"))
    
    elif isinstance(data, list):
        summary.append(f"{path}: array ({len(data)} items)")
        if data:
            # Show type of first element as representative
            summary.extend(get_type_summary(data[0], f"{path}[0]"))
            if len(data) > 1:
                # Check if all elements are same type
                types = set(type(item).__name__ for item in data)
                if len(types) > 1:
                    summary.append(f"{path}: mixed types: {types}")
    
    else:
        type_name = type(data).__name__
        summary.append(f"{path}: {type_name}")
    
    return summary


def read_json_file(filepath):
    """Read and display JSON file structure and contents."""
    path = Path(filepath)
    
    if not path.exists():
        print(f"Error: File '{filepath}' not found")
        return None
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    print("=" * 60)
    print(f"File: {filepath}")
    print("=" * 60)
    
    print("\n--- STRUCTURE SUMMARY ---\n")
    for line in get_type_summary(data):
        print(line)
    
    print("\n--- CONTENTS ---\n")
    print_structure(data)
    
    return data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python json_reader.py <path_to_json_file>")
        print("\nExample: python json_reader.py data.json")
        sys.exit(1)
    
    read_json_file(sys.argv[1])
