import json
import os
import csv


def next_name(folderpath, pattern, start=1):
    """
    Function finds next free pattern-structured name in path folderpath.
    Pattern should have one integer field, f.e. "model-{0}"
    :return: path to the next free name in the folderpath folder
    """
    counter = start
    while os.path.exists(os.path.join(folderpath, pattern.format(counter))):
        counter += 1
    return pattern.format(counter)


def json_dump(path, filename, data):
    filepath = os.path.join(path, filename)
    with open(filepath, "w") as file:
        file.write(json.dumps(data, indent=2))


def csv_dump(path, filename, data):
    filepath = os.path.join(path, filename)
    with open(filepath, 'w') as csv_file:
        all_keys = set().union(*(d.keys() for d in data))
        writer = csv.DictWriter(csv_file, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(data)


def txt_dump(path, filename, data):
    filepath = os.path.join(path, filename)
    with open(filepath, "w") as file:
        print(data, file=file)
