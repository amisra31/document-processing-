import boto3
import json
from PIL import Image
import base64
import io
import openai

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

# === STEP 3: Call GPT to Clean & Format ===
def call_openai(image_base64, ocr_text):
    prompt = f"""
You are a highly skilled assistant for parsing handwritten utility meter replacement slips.

You will receive:
1. OCR output from a handwriting-specialized model (TrOCR).
2. A scanned image of the document.
Your task is to extract structured fields accurately using both these sources.

📄 Document Layout Context:
- This is a printed meter replacement form with typed field labels and handwritten values.
- Most handwritten values are to the right of printed field names.
- There are two main sections: Old Meter and Smart Meter details.
- Fields to extract include consumer information, meter readings, and signatures.

🧠 Multi-Step Process:
1. **Interpret the OCR**: Extract all readable fields and values from the OCR text.
2. **Validate with Image**: Use the image layout and handwriting patterns to verify and fill in missing or uncertain fields.
3. **Correct & Complete**: Ensure formatting is consistent, fix OCR mistakes (e.g. `0.1` vs `01`), and complete partially missing fields where confidently possible.

📦 Output Format:
Return your answer in this structured JSON format:

{{
  "consumer_name": "",
  "consumer_address": "",
  "consumer_account_no": "",
  "consumer_mobile_no": "",
  "date_of_installation": "",
  "smart_meter_replacement_slip_no": "",
  "old_meter": {{
    "meter_no": "",
    "meter_reading": "",
    "manufacturer": "",
    "md_kw": "",
    "status": ""
  }},
  "smart_meter": {{
    "meter_no": "",
    "manufacturer": "",
    "status": ""
  }},
  "feeder_name": "",
  "subdivision": "",
  "book_no": "",
  "sl_no": "",
  "dtr": "",
  "consumer_signature": "",
  "agency_supervisor_name": "",
  "agency_supervisor_mobile": ""
}}

OCR Text:
---
{ocr_text}
---

📌 Rules:
- If a field is illegible or not present, return an empty string ("").
- Use the image only when OCR is unclear or missing.
- Avoid hallucinating values — rely on the OCR or visual cues in the form layout.
- Do not change field names or JSON keys.

Return only the final valid JSON object, with no explanation.
"""

    openai.api_key = "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"  # 🔐 Replace with your key

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You extract and clean structured information from scanned documents."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1500
    )

    return response["choices"][0]["message"]["content"]

# === MAIN PIPELINE ===
if __name__ == "__main__":
    image_path = "WhatsApp Image 2025-04-22 at 11.46.35 PM.jpeg"  # 🔁 Replace with your actual image path

    # Load + Encode Image
    image_bytes, image_base64 = load_and_encode_image(image_path)

    # Run Textract (requires AWS credentials)
    textract_kv = extract_text_with_textract(image_bytes)

    # Send to GPT to clean & fill
    final_json = call_openai(image_base64, textract_kv)

    print("\n✅ Final Structured JSON Output:\n")
    print(final_json)
