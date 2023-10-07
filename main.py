import requests
import json
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_and_extract_info(model_id: str):
    # Making a POST request
    url = f"https://civitai.com/api/v1/models/{model_id}"
    logging.info(f"Sending POST request to: {url}")
    response = requests.post(url)
    data = response.json()

    # Extracting the required info
    tags = data.get("tags", [])
    nsfw = data.get("nsfw", None)
    description = data.get("description", "")

    logging.info("Successfully fetched data from the server.")
    return tags, nsfw, description


def fetch_and_process_model_data(type, api_url, page, limit=100, sort_order="Highest Rated"):
    logging.info(f'Starting to fetch information from page {page}')
    params = {
        "limit": str(limit),
        "types": type,
        "sort": sort_order,
        "page": page
    }
    headers = {
        "Content-Type": "application/json"
    }

    response = requests.get(api_url, params=params, headers=headers)
    model_info_list = []  # Initialize the list outside the loop

    if response.status_code == 200:
        try:
            response_json = response.json()
            items = response_json.get('items', [])
            for item in items:
                main_model_name = item.get('name', '').lower()
                model_versions = item.get('modelVersions', [])
                
                for model_version in model_versions:
                    files_list = model_version.get('files', [])
                    if not files_list:
                        logging.warning('No files found for model version')
                        continue
                    
                    files = files_list[0]
                    format_ = files.get('metadata', {}).get('format', '')
                    if format_ != "Other":
                        url = files.get('downloadUrl', '')
                        date_created = files.get('createdAt', '')
                        model_id = files.get('id', '')
                        trigger_words = files.get('trainedWords', '')
                        full_model_name = files.get('name', '')
                        model_name = full_model_name.split(".")[0].lower()
                        revision = files.get('metadata', {}).get('fp', '')
                        base_model = model_version.get('baseModel', '').replace(" ", "-").replace("_", "-").lower()
                        format_ext = full_model_name.split(".")[-1]
                        
                        for image in model_version.get('images', []):
                            image_url = image.get('url', '')
                            break
                        else:
                            image_url = ''
                        
                        base_model_type_value = model_version.get('baseModelType')
                        base_model_type = (base_model_type_value or '').replace(" ", "-").replace("_", "-").lower()
                        size = files.get('metadata', {}).get('size', '')
                        size_kb = files.get('sizeKB', '')
                        tags, nsfw, description = fetch_and_extract_info(model_version['modelId'])

                        model_info = {
                            "date_created": model_version['createdAt'],
                            "model_id": model_version['modelId'],
                            "url": url,
                            "model_name": model_name,
                            "format_ext": format_ext,
                            "revision": revision,
                            "image_url": image_url,
                            "base_model": base_model,
                            "base_model_type": base_model_type,
                            "trigger_words": model_version['trainedWords'],
                            "download_count": model_version['stats']['downloadCount'],
                            "rating_count": model_version['stats']['ratingCount'],
                            "rating": model_version['stats']['rating'],
                            "full_or_pruned": size,
                            "size_kb": size_kb,
                            "tags": tags,
                            "nsfw": nsfw,
                            "description": description,
                            "model_type": type  # Added model type to the dictionary
                        }
                        model_info_list.append(model_info)
        except KeyError as e:
            logging.warning(f'KeyError: {e}')

        # Create DataFrame from model_info_list
        df = pd.DataFrame(model_info_list)
        
        # Check if file exists, if yes, read and concatenate, if not, save as new
        try:
            existing_df = pd.read_csv('model_data.csv')
            final_df = pd.concat([existing_df, df], ignore_index=True)
        except FileNotFoundError:
            final_df = df
        
        # Save to CSV
        final_df.to_csv('model_data.csv', index=False)
        
        logging.info(f'Finished processing information from page {page}')
        return model_info_list  # Returning the list of model info dictionaries for further use if needed
    else:
        logging.error(f'Failed to retrieve data from page {page}: {response.status_code}')
        return None  # Returning None in case of a failure

# Usage:
types = ["Checkpoint", "TextualInversion", "Hypernetwork", "AestheticGradient", "LORA", "Controlnet"]
api_url = "https://civitai.com/api/v1/models"
page = 1  # Assuming you want to start with page 1

for type in types:
    logging.info(f'Starting to fetch information from type {type}')
    model_data = fetch_and_process_model_data(type, api_url, page)
    logging.info(f'Finished processing information from type {type}')
