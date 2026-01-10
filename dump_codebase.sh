#!/usr/bin/env bash

# Script to dump relevant code files from the project into a single text file
# Excludes node_modules, __pycache__, .git directories, and build artifacts

OUTPUT_FILE="codebase_dump.md"

# Clear or create the output file
echo "Creating code dump in $OUTPUT_FILE"
> "$OUTPUT_FILE"

# Function to generate a directory tree structure of relevant files
generate_directory_tree() {
    echo "Generating directory tree..."
    
    echo -e "================================================================================\n" >> "$OUTPUT_FILE"
    echo -e "PROJECT DIRECTORY STRUCTURE\n" >> "$OUTPUT_FILE"
    echo -e "================================================================================\n" >> "$OUTPUT_FILE"
    
    # Create temporary files to collect all relevant files and processed directories
    TEMP_FILE=$(mktemp)
    PROCESSED_DIRS=$(mktemp)
    
    # Add root directory to processed dirs
    echo "." > "$PROCESSED_DIRS"
    
    # Find all relevant files using the same patterns as for file content
    # Python files
    find . -type f -name "*.py" \
        -not -path "*/\.*" \
        -not -path "*/__pycache__/*" \
        -not -path "*/venv/*" \
        -not -path "*/env/*" \
        -not -path "*/node_modules/*" \
        -not -path "*/ios/*" \
        -not -path "*/android/*" \
        >> "$TEMP_FILE"
    
    # JavaScript/TypeScript files
    find . -type f \( -name "*.js" -o -name "*.jsx" -o -name "*.ts" -o -name "*.tsx" \) \
        -not -path "*/\.*" \
        -not -path "*/node_modules/*" \
        -not -path "*/dist/*" \
        -not -path "*/build/*" \
        -not -path "*/ios/*" \
        -not -path "*/android/*" \
        >> "$TEMP_FILE"
    
    # HTML/CSS files
    find . -type f \( -name "*.html" -o -name "*.css" -o -name "*.scss" -o -name "*.sass" \) \
        -not -path "*/\.*" \
        -not -path "*/node_modules/*" \
        -not -path "*/dist/*" \
        -not -path "*/build/*" \
        -not -path "*/ios/*" \
        -not -path "*/android/*" \
        >> "$TEMP_FILE"
    
    # Configuration files
    find . -type f \( -name "*.json" -o -name "*.yml" -o -name "*.yaml" -o -name "*.xml" -o -name "*.toml" -o -name "*.ini" -o -name "*.conf" -o -name "*.config" \) \
        -not -path "*/\.*" \
        -not -path "*/node_modules/*" \
        -not -path "*/dist/*" \
        -not -path "*/build/*" \
        -not -path "*/__pycache__/*" \
        -not -path "*/ios/*" \
        -not -path "*/android/*" \
        >> "$TEMP_FILE"
    
    # Important files with no extension at root level
    find . -maxdepth 1 -type f -not -path "*/\.*" \
        -not -name "$OUTPUT_FILE" \
        -not -path "*/ios/*" \
        -not -path "*/android/*" \
        >> "$TEMP_FILE"
    
    # Sort the files and process them to create a hierarchical tree
    # Add root directory first
    echo -e "+-- ." >> "$OUTPUT_FILE"
    
    sort -u "$TEMP_FILE" | while read file; do
        # Remove ./ prefix if it exists
        file="${file#./}"
        
        # Get just the filename (last part)
        filename=$(basename "$file")
        
        # Get the directory path
        dirpath=$(dirname "$file")
        
        # Process directory hierarchy
        if [ "$dirpath" != "." ]; then
            # Split the dirpath into components and build paths incrementally
            parts=$(echo "$dirpath" | tr '/' ' ')
            current_path=""
            current_depth=0
            
            for part in $parts; do
                # Build the current path
                if [ -z "$current_path" ]; then
                    current_path="$part"
                else
                    current_path="$current_path/$part"
                fi
                
                # Check if this directory has been processed already
                if ! grep -q "^$current_path$" "$PROCESSED_DIRS"; then
                    # Calculate indentation based on depth
                    indent=""
                    for ((i=0; i<current_depth; i++)); do
                        indent="${indent}|   "
                    done
                    
                    # Add directory to tree with proper indentation
                    echo -e "${indent}+-- ${part}/" >> "$OUTPUT_FILE"
                    
                    # Mark as processed by adding to our temporary file
                    echo "$current_path" >> "$PROCESSED_DIRS"
                fi
                
                current_depth=$((current_depth + 1))
            done
        fi
        
        # Calculate proper indentation for the file
        file_indent=""
        dir_depth=$(echo "$dirpath" | tr -cd '/' | wc -c)
        if [ "$dirpath" != "." ]; then
            dir_depth=$((dir_depth + 1))
        else
            dir_depth=0
        fi
        
        for ((i=0; i<dir_depth; i++)); do
            file_indent="${file_indent}|   "
        done
        
        # Add the file with indentation
        echo -e "${file_indent}+-- ${filename}" >> "$OUTPUT_FILE"
    done
    
    # Add a final separator
    echo -e "\n================================================================================\n" >> "$OUTPUT_FILE"
    echo -e "FILE CONTENTS\n" >> "$OUTPUT_FILE"
    echo -e "================================================================================\n" >> "$OUTPUT_FILE"
    
    # Clean up temporary files
    rm "$TEMP_FILE" "$PROCESSED_DIRS"
}

# Function to add a file to the dump
add_file_to_dump() {
    local file="$1"
    
    # Get file size in KB
    local size_kb=$(du -k "$file" | cut -f1)
    
    # Skip files larger than 1MB (1024KB) as they're likely binary or too large
    if [ "$size_kb" -gt 1024 ]; then
        echo "Skipping large file: $file ($size_kb KB)"
        return
    fi
    
    # Check if file is likely binary
    if file "$file" | grep -q "binary"; then
        echo "Skipping binary file: $file"
        return
    fi
    
    echo "Adding file: $file"
    
    # Add file header
    echo -e "\n\n================================================================================" >> "$OUTPUT_FILE"
    echo "FILE: $file" >> "$OUTPUT_FILE"
    echo "================================================================================" >> "$OUTPUT_FILE"
    
    # Add file content
    cat "$file" >> "$OUTPUT_FILE"
}

# Generate directory tree structure at the beginning
generate_directory_tree

# Find and process Python files
find . -type f -name "*.py" \
    -not -path "*/\.*" \
    -not -path "*/__pycache__/*" \
    -not -path "*/venv/*" \
    -not -path "*/env/*" \
    -not -path "*/node_modules/*" \
    -not -path "*/ios/*" \
    -not -path "*/android/*" \
    | while read file; do
        add_file_to_dump "$file"
    done

# Find and process JavaScript/TypeScript files
find . -type f \( -name "*.js" -o -name "*.jsx" -o -name "*.ts" -o -name "*.tsx" \) \
    -not -path "*/\.*" \
    -not -path "*/node_modules/*" \
    -not -path "*/dist/*" \
    -not -path "*/build/*" \
    -not -path "*/ios/*" \
    -not -path "*/android/*" \
    | while read file; do
        add_file_to_dump "$file"
    done

# Find and process HTML/CSS files
find . -type f \( -name "*.html" -o -name "*.css" -o -name "*.scss" -o -name "*.sass" \) \
    -not -path "*/\.*" \
    -not -path "*/node_modules/*" \
    -not -path "*/dist/*" \
    -not -path "*/build/*" \
    -not -path "*/ios/*" \
    -not -path "*/android/*" \
    | while read file; do
        add_file_to_dump "$file"
    done

# Add configuration files
find . -type f \( -name "*.json" -o -name "*.yml" -o -name "*.yaml" -o -name "*.xml" -o -name "*.toml" -o -name "*.ini" -o -name "*.conf" -o -name "*.config" \) \
    -not -path "*/\.*" \
    -not -path "*/node_modules/*" \
    -not -path "*/dist/*" \
    -not -path "*/build/*" \
    -not -path "*/__pycache__/*" \
    -not -path "*/ios/*" \
    -not -path "*/android/*" \
    | while read file; do
        add_file_to_dump "$file"
    done

# Add important files with no extension at root level
find . -maxdepth 1 -type f -not -path "*/\.*" \
    -not -name "$OUTPUT_FILE" \
    -not -path "*/ios/*" \
    -not -path "*/android/*" \
    | while read file; do
        add_file_to_dump "$file"
    done

echo "Code dump completed! Output saved to $OUTPUT_FILE"
echo "Total size: $(du -h "$OUTPUT_FILE" | cut -f1)"
echo "Copying to clipboard..."
cat "$OUTPUT_FILE" | pbcopy
echo "Content copied to clipboard!"