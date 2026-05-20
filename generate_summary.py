import os

def summarize_python_files(source_dir='src', output_file='summary.txt'):
    # Check if the source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: The directory '{source_dir}' does not exist.")
        return

    with open(output_file, 'w', encoding='utf-8') as summary:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    
                    # Write a separator and the file name for clarity
                    summary.write(f"{'='*80}\n")
                    summary.write(f"FILE: {file_path}\n")
                    summary.write(f"{'='*80}\n\n")
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            summary.write(f.read())
                    except Exception as e:
                        summary.write(f"Error reading file: {e}")
                    
                    # Add spacing between files
                    summary.write("\n\n")
    
    print(f"Done! Summary created at: {output_file}")

if __name__ == "__main__":
    summarize_python_files()
