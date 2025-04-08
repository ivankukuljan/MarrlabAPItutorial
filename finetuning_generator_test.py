import pandas as pd
import json




def gpt_api_finetuning_entry(prompt_text, image_url, ground_truth_label):

    prompt_text = json.dumps(prompt_text)

    entry = '{"messages": [{"role": "system", "content": "You are an assistant that identifies cell types."}, {"role": "user", "content": ' + prompt_text + '}, {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "' + image_url + '"}}]}, {"role": "assistant", "content": "' + ground_truth_label + '"}]}'

    # jsonl_entry = json.dumps(entry)

    return entry



def generate_finetuning_jsonl(prompt_text, dataset_csv_path, subset_oline_location, fine_tuning_jsonl_path):

    dataset_csv = pd.read_csv(dataset_csv_path)

    for _, row in dataset_csv.iterrows():
        image_name = row['image_name']
        image_url = subset_oline_location + image_name + '.jpg'
        ground_truth_label = row['label']


    with open(fine_tuning_jsonl_path, 'a') as f:
        for _, row in dataset_csv.iterrows():
            image_name = row['image_name']

            image_url = subset_oline_location + image_name + '.jpg'

            ground_truth_label = row['label']             
            
            jsonl_entry = gpt_api_finetuning_entry(prompt_text, image_url, ground_truth_label)
            f.write(jsonl_entry + '\n')