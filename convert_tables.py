import os
import pandas as pd
from bs4 import BeautifulSoup

def convert_html_tables_in_markdown(input_path, output_path):
    # Read the Markdown content
    with open(input_path, "r", encoding="utf-8") as file:
        markdown_text = file.read()

    soup = BeautifulSoup(markdown_text, "html.parser")
    tables = soup.find_all("html")

    print(f"Found {len(tables)} HTML tables, converting...")

    # Store original HTML and converted Markdown tables
    replacements = []

    for i, table in enumerate(tables):
        try:
            # Parse the table into a DataFrame
            df = pd.read_html(str(table))[0]
            # Convert DataFrame to Markdown format
            markdown_table = df.to_markdown(index=False)
            # Add to the list of replacements
            replacements.append((str(table), markdown_table))
        except Exception as e:
            print(f"⚠️ Failed to parse table {i+1}, skipping: {e}")
            continue

    # Replace original HTML tables with Markdown versions
    for original, replacement in replacements:
        markdown_text = markdown_text.replace(original, f"\n\n{replacement}\n\n")

    # Write the new Markdown content to file
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(markdown_text)

    print(f"✅ Table conversion completed. Saved to: {output_path}")

# Example usage (adjust file paths as needed)
if __name__ == "__main__":
    input_file = "data/Multilingual.md"             # Original Markdown file
    output_file = "data/Multilingual_cleaned.md"    # Output cleaned Markdown file
    convert_html_tables_in_markdown(input_file, output_file)
