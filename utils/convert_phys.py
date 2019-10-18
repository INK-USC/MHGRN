import json
import re
import sys
from tqdm import tqdm

__all__ = ['convert_to_phys_statement']

# String used to indicate a blank
BLANK_STR = "___"


def convert_to_phys_statement(qa_file: str, label_file: str, output_file1: str, output_file2: str):
    print(f'converting {qa_file} to entailment dataset...')
    nrow = sum(1 for _ in open(qa_file, 'r'))
    with open(output_file1, 'w') as output_handle1, open(output_file2, 'w') as output_handle2, open(qa_file, 'r') as qa_handle, open(label_file) as label_handle:
        # print("Writing to {} from {}".format(output_file, qa_file))
        for line, label in tqdm(zip(qa_handle, label_handle), total=nrow):
            json_line = json.loads(line)
            label = label.strip()
            output_dict = convert_qajson_to_entailment(json_line, label)
            output_handle1.write(json.dumps(output_dict))
            output_handle1.write("\n")
            output_handle2.write(json.dumps(output_dict))
            output_handle2.write("\n")
    print(f'converted statements saved to {output_file1}, {output_file2}')
    print()


# Convert the QA file json to output dictionary containing premise and hypothesis
def convert_qajson_to_entailment(qa_json: dict, label: str):
    question_text = qa_json['goal']
    choice1 = qa_json['sol1']
    choice2 = qa_json['sol2']
    s1 = question_text + ' ' + choice1
    s2 = question_text + ' ' + choice2
    dic = {'answerKey': ('A' if label == '0' else 'B'),
           'id': qa_json['id'],
           'question': {'stem': question_text,
                        'choices': [{'label': 'A', 'text': choice1}, {'label': 'B', 'text': choice2}]},
           'statements': [{'label': label == '0', 'statement': s1}, {'label': label == '1', 'statement': s2}]
           }
    return dic
