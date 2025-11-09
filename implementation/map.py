import os 

def generate_ordered_class_names(winds_file_path, words_file_path):
    

    n_id_to_name_map = {}
    with open(words_file_path, 'r')as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                n_id_to_name_map[parts[0]] = parts[1]

    ordered_n_ids = []
    with open(winds_file_path, 'r') as f:
    
        for line in f:
            n_id = line.strip()
            if n_id:
                ordered_n_ids.append(n_id)


    if len(ordered_n_ids) != 200:
        print(f"Warning: Expected 200 class IDs, but found {len(ordered_n_ids)}")

    
    final_class_names = []
    for n_id in ordered_n_ids:
        if n_id in n_id_to_name_map:
            name = n_id_to_name_map[n_id]
            final_class_names.append(name)
        else:
            final_class_names.append(f"name not found for {n_id}")

    return final_class_names


WORDS_PATH = 'words.txt'
WINDS_PATH = 'wnids.txt' 

CLASS_NAMES_LIST = generate_ordered_class_names(WINDS_PATH, WORDS_PATH)

print(f"Loaded {len(CLASS_NAMES_LIST)} class names.")



#create a file to store the mapped id and class
with open('class_names_mapping.txt', 'w') as f:
    for idx, class_name in enumerate(CLASS_NAMES_LIST):
        f.write(f"{idx}\t{class_name}\n")