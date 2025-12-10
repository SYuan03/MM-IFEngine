import argparse
import json
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--jsonl_path', type=str, required=True)
args = parser.parse_args()

def sorted_csv_gen(jsonl_path, csv_path, key):
    data = []
    with open(jsonl_path, 'r') as jsonl_file:
        for line in jsonl_file:
            entry = json.loads(line)
            data.append(entry)

    data.sort(key=lambda x: list(x.values())[0], reverse=True)

    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['filename', key])  # Header row
        for entry in data:
            filename, value = list(entry.items())[0]
            writer.writerow([filename, value])

jsonl_path = args.jsonl_path
csv_path = jsonl_path.replace('.jsonl', '_sorted.csv')
key = 'score'
sorted_csv_gen(jsonl_path, csv_path, key)