#!/usr/bin/env python3
"""
Theme Counter for TrueNorth Test Cases
Counts the number of examples for each theme in test_cases.json and saves as CSV.
"""

import json
import csv
from collections import Counter
import os

def count_themes_from_json(json_file_path, output_csv_path):
    """
    Count themes from test_cases.json and save results to CSV.
    
    Args:
        json_file_path (str): Path to the test_cases.json file
        output_csv_path (str): Path for the output CSV file
    """
    
    try:
        # Read the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as file:
            test_cases = json.load(file)
        
        print(f"Loaded {len(test_cases)} test cases from {json_file_path}")
        
        # Extract and count themes
        theme_counter = Counter()
        
        for case in test_cases:
            if 'theme' in case:
                # Handle themes that might have multiple values separated by commas
                theme = case['theme'].strip()
                # Remove trailing comma if present
                if theme.endswith(','):
                    theme = theme[:-1]
                
                # Split by comma and count each theme separately if multiple
                themes = [t.strip() for t in theme.split(',') if t.strip()]
                
                for individual_theme in themes:
                    theme_counter[individual_theme] += 1
        
        # Sort themes by count (descending) then alphabetically
        sorted_themes = sorted(theme_counter.items(), 
                             key=lambda x: (-x[1], x[0]))
        
        # Write to CSV
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['Theme', 'Count', 'Percentage'])
            
            # Calculate total for percentages
            total_examples = sum(theme_counter.values())
            
            # Write data rows
            for theme, count in sorted_themes:
                percentage = (count / total_examples) * 100
                writer.writerow([theme, count, f"{percentage:.1f}%"])
            
            # Write summary row
            writer.writerow(['TOTAL', total_examples, '100.0%'])
        
        print(f"\nTheme analysis saved to: {output_csv_path}")
        print(f"Total themes found: {len(theme_counter)}")
        print(f"Total examples: {total_examples}")
        
        # Display results
        print("\n" + "="*50)
        print("THEME DISTRIBUTION")
        print("="*50)
        print(f"{'Theme':<25} {'Count':<8} {'Percentage'}")
        print("-"*50)
        
        for theme, count in sorted_themes:
            percentage = (count / total_examples) * 100
            print(f"{theme:<25} {count:<8} {percentage:>8.1f}%")
        
        print("-"*50)
        print(f"{'TOTAL':<25} {total_examples:<8} {'100.0%':>8}")
        
        return theme_counter
        
    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{json_file_path}': {e}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    """Main function to run the theme counter."""
    
    # Define file paths
    json_file = "test_cases.json"
    csv_file = "theme_distribution.csv"
    
    # Check if input file exists
    if not os.path.exists(json_file):
        print(f"Input file '{json_file}' not found in current directory.")
        print("Please ensure test_cases.json is in the same directory as this script.")
        return
    
    # Count themes and generate CSV
    theme_counts = count_themes_from_json(json_file, csv_file)
    
    if theme_counts:
        print(f"\n✓ Analysis complete! Check '{csv_file}' for detailed results.")
    else:
        print("\n✗ Analysis failed. Please check the input file and try again.")

if __name__ == "__main__":
    main()