# This is a sample Python script.
import io
import json

from src.extract_h5p import convert_h5p_folder_to_instruction_pairs, INPUT_DIR, OUTPUT_FILE


# Press Strg+F5 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def main():
   convert_h5p_folder_to_instruction_pairs(INPUT_DIR, OUTPUT_FILE)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
   main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
