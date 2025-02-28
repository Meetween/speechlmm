"""
This script processes a JSONL file containing data about audio files and their transcriptions, prompts, and AI-generated texts. It reformats the data, extracting instructions and outputs from the generated text. The extracted information is used to create new JSON objects, which are then written to a new JSONL file.

For each JSON object in the input file, the script creates a new JSON object with the following keys:
- id (int): unique identifier for the audio file
- audio (str): path to the audio file
- conversation (list of dict): list of dictionaries containing the following keys
    - instruction (str): instruction generated from the AI model
    - output (str): instruction-following output generated by the AI model

Given the AI generated text, the extraction rules for instruction and output are the followings:

- Generated text format:
- "Q:<question1>A:<answer1>Q:<question2>A:...and so on"

clean question and answer pairs removing the "Q:" and "A:" prefixes, new line characters and trailing spaces or quotes.

Exaple of conversation given the generated text above:
conversation = [
    {
        "instruction": "<question1>",
        "output": "<answer1>"
    },
    {
        "instruction": "<question2>",
        "output": "<answer2>"
    },
    ...
]

From the generated text, the script extracts the instruction and output pairs. The instruction is the question, and the output is the answer.
The script then writes the extracted instruction and output pairs to the output file in JSONL format.
"""

import argparse
import json
import random


def process_data(input_file, output_file):
    with open(input_file, "r") as f:
        data = f.readlines()

    new_data = []
    for line in data:
        json_obj = json.loads(line)

        generated_text = json_obj["generated_text"]

        # Extract instructions and outputs from the generated text

        # Split the generated text into question-answer pairs

        # Remove the "Q:" and "A:" prefixes, new line characters, and trailing spaces or quotes
        conversation = []
        # "Q:<question1>A:<answer1>Q:<question2>A:...and so on"
        # check format
        if "Q:" not in generated_text:
            print(f"Skipping line: {json_obj['audio']}")
            continue
        text = generated_text.split("Q:", 1)[1]
        while "Q:" in text:
            # extract question and answer
            question, text = text.split("A:", 1)

            answer = text.split("Q:", 1)[0]
            text = text.split("Q:", 1)[1]
            # remove new line characters
            question = question.replace("\n", "")
            answer = answer.replace("\n", "")
            # remove trailing spaces or quotes
            question = question.strip().strip('"')
            answer = answer.strip().strip('"')

            conversation.append({"instruction": question, "output": answer})
        # last question and answer
        question, answer = text.split("A:", 1)
        question = question.replace("\n", "")
        answer = answer.replace("\n", "")
        question = question.strip().strip('"')
        answer = answer.strip().strip('"')
        conversation.append({"instruction": question, "output": answer})
        new_json_obj = create_json_object(json_obj, conversation)
        new_data.append(new_json_obj)

    with open(output_file, "w") as f:
        for line in new_data:
            f.write(json.dumps(line) + "\n")


def create_json_object(json_obj, conversation):
    new_json_obj = {
        "id": json_obj["id"],
        "audio": json_obj["audio"],
        "conversations": conversation,
    }
    return new_json_obj


# main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process AI generated data")
    parser.add_argument("-i", type=str, help="Path to the input file")
    parser.add_argument("-o", type=str, help="Path to the output file")
    args = parser.parse_args()

    process_data(args.i, args.o)


# Usage
# python process.py -i test.jsonl -o output.jsonl
