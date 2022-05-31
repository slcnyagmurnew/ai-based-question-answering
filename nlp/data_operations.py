import json
import csv


# divide json into json files via selected fields
def create_json(file, fields, json_name):
    f = open(file, "r")
    json_object = json.load(f)
    f.close()

    new_file = open(f"data/{json_name}.json", "w")
    json_list = []

    for q in json_object:
        for f in fields:
            if f in q['question'].lower():
                labeled_obj = update_json_object(q, json_name)
                json_list.append(labeled_obj)

    json.dump(json_list, new_file, indent=4)
    new_file.close()


# add label field into all json objects, color etc.
def update_json_object(obj, label):
    obj["label"] = label
    return obj


# merge all json files into one json file
def merge_jsons(files, new_json):
    merged_json = open(f'data/{new_json}', 'w')
    merged_list = []

    for f in files:
        with open(f'data/{f}.json', 'r') as file:
            json_obj = json.load(file)
            for j in json_obj:
                merged_list.append(j)

        file.close()
    json.dump(merged_list, merged_json, indent=4)
    merged_json.close()


# from train json
def create_csv(json_file, filename):
    # field names
    fields = ['Question', 'Label']
    # data rows of csv file
    rows = []
    # name of csv file

    f = open(f'data/{json_file}', 'r')
    json_object = json.load(f)
    for j in json_object:
        rows.append([j['question'], j['label']])
    f.close()

    # writing to csv file
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        # writing the fields
        csvwriter.writerow(fields)
        # writing the data rows
        csvwriter.writerows(rows)
    csvfile.close()


# from test json
def append_csv(json_file, filename):
    # Open our existing CSV file in append mode
    # Create a file object for this file
    jf = open(f'data/{json_file}', 'r')
    f = open(f'data/{filename}', 'a')
    json_object = json.load(jf)
    # Pass this file object to csv.writer()
    # and get a writer object
    writer_object = csv.writer(f)
    for j in json_object:
        # Pass the list as an argument into
        # the writerow()
        writer_object.writerow([j['question'], j['label']])
    # Close the file object
    f.close()


if __name__ == '__main__':
    # create_json("data/test.json", ['color', 'colour'], 'color')
    # create_json("data/test.json", ['what is this', 'what is it'], 'object')
    # create_json("data/test.json", ['how many', 'how much'], 'number')
    # merge_jsons(['color', 'object', 'number'], 'test0.json')
    # create_csv('train0.json', 'data/train_questions.csv')
    append_csv('test0.json', 'train_questions.csv')
