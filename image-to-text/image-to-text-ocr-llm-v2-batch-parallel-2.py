import boto3
import json
from PIL import Image
import base64
import io
import openai
import cv2
import numpy as np
#import torch
#from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

DEFAULT_OUTPUT = {
    "consumer_name": "",
    "consumer_address": "",
    "consumer_account_no": "",
    "consumer_mobile_no": "",
    "date_of_installation": "",
    "smart_meter_replacement_slip_no": "",
    "feeder_name": "",
    "subdivision": "",
    "book_no": "",
    "sl_no": "",
    "dtr": "",
    "consumer_signature": "",
    "agency_supervisor_name": "",
    "agency_supervisor_mobile": "",
    "division_signature": "",
    "old_meter": {
        "meter_no": "",
        "meter_reading": "",
        "manufacturer": "",
        "meter_md_kw": "",
        "box_seal_no": "",
        "nic_seal_no": "",
        "body_seal_no": "",
        "terminal_seal_no": "",
        "meter_status": "",
        "consumer_category": "",
        "lat_long": ""
    },
    "smart_meter": {
        "meter_no": "",
        "meter_reading": "",
        "manufacturer": "",
        "meter_md_kw": "",
        "box_seal_no": "",
        "nic_seal_no": "",
        "body_seal_no": "",
        "terminal_seal_no": "",
        "meter_status": "",
        "consumer_category": "",
        "lat_long": ""
    }
}



# === STEP 1: Load and Encode Image ===
def load_and_encode_image(image_path):
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    return image_bytes, image_base64



# === STEP 2: Call AWS Textract ===
def extract_text_with_textract(image_bytes):
    client = boto3.client("textract", region_name="us-east-1")  # Update region if needed

    response = client.analyze_document(
        Document={"Bytes": image_bytes},
        FeatureTypes=["FORMS"]
    )

    key_map = {}
    value_map = {}
    block_map = {}

    for block in response['Blocks']:
        block_id = block['Id']
        block_map[block_id] = block
        if block['BlockType'] == "KEY_VALUE_SET":
            if 'KEY' in block.get('EntityTypes', []):
                key_map[block_id] = block
            else:
                value_map[block_id] = block

    def get_text(result, blocks_map):
        text = ''
        if 'Relationships' in result:
            for rel in result['Relationships']:
                if rel['Type'] == 'CHILD':
                    for child_id in rel['Ids']:
                        word = blocks_map[child_id]
                        if word['BlockType'] == 'WORD':
                            text += word['Text'] + ' '
                        elif word['BlockType'] == 'SELECTION_ELEMENT':
                            if word['SelectionStatus'] == 'SELECTED':
                                text += 'X '
        return text.strip()

    field_map = {}
    for key_block in key_map.values():
        key_text = get_text(key_block, block_map)
        for rel in key_block.get('Relationships', []):
            if rel['Type'] == 'VALUE':
                for value_id in rel['Ids']:
                    value_block = value_map.get(value_id)
                    value_text = get_text(value_block, block_map)
                    field_map[key_text] = value_text

    return field_map


    def format_field_map_as_prompt(field_map):
    prompt_lines = ["I have extracted the following fields from the meter slip using OCR:"]
    for key, value in field_map.items():
        # Recursively format nested fields
        if isinstance(value, dict):
            prompt_lines.append(f"\n{key}:")
            for sub_key, sub_val in value.items():
                prompt_lines.append(f"  - {sub_key.replace('_', ' ').title()}: {sub_val or '[Missing]'}")
        else:
            prompt_lines.append(f"- {key.replace('_', ' ').title()}: {value or '[Missing]'}")

    prompt_lines.append("\nPlease verify this against the form image and extract the correct values in JSON format as per the schema.")
    return "\n".join(prompt_lines)



def call_openai(image_base64, ocr_text):
    openai.api_key = "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"

    schema = {
        "name": "extract_form_data",
        "description": "Extract structured fields from scanned handwritten meter replacement slips",
        "parameters": {
            "type": "object",
            "properties": {
                "consumer_name": {"type": "string"},
                "consumer_address": {"type": "string"},
                "consumer_account_no": {"type": "string"},
                "consumer_mobile_no": {"type": "string"},
                "date_of_installation": {"type": "string"},
                "smart_meter_replacement_slip_no": {"type": "string"},
                "feeder_name": {"type": "string"},
                "subdivision": {"type": "string"},
                "book_no": {"type": "string"},
                "sl_no": {"type": "string"},
                "dtr": {"type": "string"},
                "consumer_signature": {"type": "string"},
                "agency_supervisor_name": {"type": "string"},
                "agency_supervisor_mobile": {"type": "string"},
                "division_signature": {"type": "string"},
                "old_meter": {
                    "type": "object",
                    "properties": {
                        "meter_no": {"type": "string"},
                        "meter_reading": {"type": "string"},
                        "manufacturer": {"type": "string"},
                        "meter_md_kw": {"type": "string"},
                        "box_seal_no": {"type": "string"},
                        "nic_seal_no": {"type": "string"},
                        "body_seal_no": {"type": "string"},
                        "terminal_seal_no": {"type": "string"},
                        "meter_status": {"type": "string"},
                        "consumer_category": {"type": "string"},
                        "lat_long": {"type": "string"}
                    }
                },
                "smart_meter": {
                    "type": "object",
                    "properties": {
                        "meter_no": {"type": "string"},
                        "meter_reading": {"type": "string"},
                        "manufacturer": {"type": "string"},
                        "meter_md_kw": {"type": "string"},
                        "box_seal_no": {"type": "string"},
                        "nic_seal_no": {"type": "string"},
                        "body_seal_no": {"type": "string"},
                        "terminal_seal_no": {"type": "string"},
                        "meter_status": {"type": "string"},
                        "consumer_category": {"type": "string"},
                        "lat_long": {"type": "string"}
                    }
                }
            },
            "required": []
        }
    }

    response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": """You are a highly skilled assistant for parsing handwritten utility meter replacement slips.

            You will receive:
            1. OCR output from a handwriting-specialized model (Textract).
            2. A scanned image of the document.

            Your task is to extract structured fields accurately using both these sources.

            Document Layout Context:
            - This is a printed meter replacement form with typed field labels and handwritten values.
            - Most handwritten values are to the right of printed field names.
            - There are three main sections: Account No, Old Meter and Smart Meter details.
            
            Multi-Step Process:
            1. **Interpret the OCR**: Extract all readable fields and values from the OCR text.
            2. **Validate with Image**: Use the image layout and handwriting patterns to verify and fill in missing or uncertain fields.
            3. **Correct & Complete**: Ensure formatting is consistent, fix OCR mistakes (e.g. `0.1` vs `01`), and complete partially missing fields where confidently possible.

            Output Format:
            Return your answer in this structured JSON format by calling the tool function."""
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": format_field_map_as_prompt(ocr_text)},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]
        }
    ],
    tools=[{"type": "function", "function": schema}],
    tool_choice={"type": "function", "function": {"name": "extract_form_data"}}
)

    )
    print("GPT raw tool_calls:", response["choices"][0].get("tool_calls"))
    print("GPT full message:", response["choices"][0])

    tool_calls = response["choices"][0]["message"].get("tool_calls", [])
    if tool_calls:
        arguments = tool_calls[0]["function"]["arguments"]
        return json.loads(arguments)
    else:
        return {"error": "No structured tool output"}


def merge_with_defaults(output, defaults):
    result = deepcopy(defaults)
    for key, value in output.items():
        if isinstance(value, dict) and key in result:
            result[key] = merge_with_defaults(value, result[key])
        else:
            result[key] = value
    return result




# === PROCESS A SINGLE IMAGE FILE ===
def process_image_file(image_path):
    image_bytes, image_base64 = load_and_encode_image(image_path)
    
    #textract_bytes, image_base64 = enhance_image(image_path)
    textract_kv = extract_text_with_textract(image_bytes)

    # GPT now returns structured JSON directly
    structured_output = call_openai(image_base64, textract_kv)

    structured_output = merge_with_defaults(structured_output, DEFAULT_OUTPUT)

    # Add filename for tracking
    structured_output["image_file_name"] = os.path.basename(image_path)
    return structured_output


# === MAIN BATCH PIPELINE ===
if __name__ == "__main__":
    image_folder = "./input_images/sample"  # Folder containing batch images
    image_paths = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    output_data = []

    print(f"Starting parallel processing of {len(image_paths)} images...\n")

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_image_file, path): path for path in image_paths}
        for future in as_completed(futures):
            result = future.result()
            output_data.append(result)
            print(f"Processed: {result.get('image_file_name', 'unknown')}")

    with open("batch_output_sample.json", "w") as outfile:
        json.dump(output_data, outfile, indent=2)

    print("\nBatch processing complete. Output saved to 'batch_output.json'.")