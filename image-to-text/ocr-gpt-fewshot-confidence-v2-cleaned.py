import boto3
import json
import base64
import openai
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import logging
from typing import Dict, List, Any, Tuple, Optional
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration constants

CONFIG = {
    "confidence_threshold": 85,
    "textract_region": "us-east-1",
    "max_workers": 5,
    "output_file": f"batch_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    "example_image_path": "./WhatsApp Image 2025-04-22 at 11.44.12 PM.jpeg",
    "input_folder": "./input_images",
    "supported_extensions": (".jpg", ".jpeg", ".png")
}
# Default output structure
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
    },
    "overall_confidence": 0.0
}

# Example data
with open('example_data.json', 'w') as f:
    json.dump({
        "ocr_text": """  - Consumer Mobile No. :: 6000404901
                        - Book No.:: 031
                        - Consumer Account No.: 161000015278
                        - Consumer Signature;: Tularan limge
                        - Consumer Name: TULA RAM LIMBU
                        - Smart Meter Replacement Slip No.: 567070
                        - Feeder Name:: DAYANG
                        - SL No.:: [Missing]
                        - Division/Sub-Division Authority Signature:: [Missing]
                        - Date of Installation and Time: 04-03-2025
                        - Consumer Address :: LAMBAPATHAR
                        - they Supervisor Mobile No.: [Missing]
                        - rocy Supervisor Name.: [Missing]
                        - Sub- Division: KHERONI
                        - DTR: 103

                        Additionally, here is the tabular data extracted from the slip (used for old/smart meter details):

                        Table 1:
                        Details | Old Meter Details | Smart Meter Details
                        1. Meter No. | GE887158 | AIK 248431
                        2 Meter Reading (KWH/KVAH) | 0 | 0
                        Meter Manufacture | GENUS | KIMBAL
                        Meter MD(KW) | 0.49 | -
                        Box Seal No. |  | 9611505
                        Meter NIC Seal No. |  | 
                        Body Seal No. |  | 9611506
                        Terminal Seal No. |  | 0
                        Meter Status | FAULTY METER | OK
                        0. Consumer Category | JEEVAN DHARA (DOM) | 
                        1. Lat/ Long |  | """,
        "json_output": {
                "consumer_name": "TUKA RAM LIMBU",
                "consumer_address": "LAMBADPATHAR",
                "consumer_account_no": "161000015278",
                "consumer_mobile_no": "6000404901",
                "date_of_installation": "04-03-2025",
                "smart_meter_replacement_slip_no": "567070",
                "feeder_name": "DAYANG",
                "subdivision": "KHERONI",
                "book_no": "031",
                "sl_no": "",
                "dtr": "123",
                "consumer_signature": "Telaga Limse",
                "agency_supervisor_name": "",
                "agency_supervisor_mobile": "",
                "division_signature": "",
                "old_meter": {
                    "meter_no": "GE887158",
                    "meter_reading": "0",
                    "manufacturer": "GENUS",
                    "meter_md_kw": "0.49",
                    "box_seal_no": "",
                    "nic_seal_no": "",
                    "body_seal_no": "",
                    "terminal_seal_no": "",
                    "meter_status": "FAULTY METER",
                    "consumer_category": "JEEVAN DHARA (DOM)",
                    "lat_long": ""
                },
                "smart_meter": {
                    "meter_no": "AIK248431",
                    "meter_reading": "0",
                    "manufacturer": "KIMBAL",
                    "meter_md_kw": "",
                    "box_seal_no": "9611505",
                    "nic_seal_no": "",
                    "body_seal_no": "9611506",
                    "terminal_seal_no": "",
                    "meter_status": "OK",
                    "consumer_category": "",
                    "lat_long": ""
                }
        }
    }, f)


class APIClients:
    """Singleton class to manage API clients"""
    
    _textract_client = None
    
    @classmethod
    def get_textract_client(cls):
        if cls._textract_client is None:
            cls._textract_client = boto3.client("textract", region_name=CONFIG["textract_region"])
        return cls._textract_client


# === Image Handling ===
def encode_image(image_path: str) -> Tuple[bytes, str]:
    """Load image from path and return both raw bytes and base64 encoded string"""
    try:
        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        return image_bytes, image_base64
    except Exception as e:
        logger.error(f"Error encoding image {image_path}: {str(e)}")
        raise


# === Textract OCR Processing ===
def extract_text_blocks(result: Dict, block_map: Dict, include_confidence: bool = False) -> Tuple[str, Optional[float]]:
    """Extract text from block relationships with optional confidence scores"""
    text = ""
    confidences = []
    
    if "Relationships" in result:
        for rel in result["Relationships"]:
            if rel["Type"] == "CHILD":
                for child_id in rel["Ids"]:
                    word = block_map[child_id]
                    if word["BlockType"] == "WORD":
                        text += word["Text"] + " "
                        if include_confidence:
                            confidences.append(word.get("Confidence", 0))
                    elif word["BlockType"] == "SELECTION_ELEMENT" and word["SelectionStatus"] == "SELECTED":
                        text += "X "
                        if include_confidence:
                            confidences.append(word.get("Confidence", 0))
    
    text = text.strip()
    
    if include_confidence and confidences:
        avg_conf = round(sum(confidences) / len(confidences), 2)
        return text, avg_conf
    
    return text, None if include_confidence else None


def calculate_average_confidence(scores: List[float]) -> Optional[float]:
    """Calculate average confidence from a list of scores"""
    return round(sum(scores) / len(scores), 2) if scores else None


def extract_text_with_textract(image_bytes: bytes, confidence_threshold: int = CONFIG["confidence_threshold"]) -> Dict:
    """Process image through AWS Textract for OCR with forms and tables extraction"""
    try:
        client = APIClients.get_textract_client()
        response = client.analyze_document(
            Document={"Bytes": image_bytes},
            FeatureTypes=["FORMS", "TABLES"]
        )
    except Exception as e:
        logger.error(f"Textract API error: {str(e)}")
        raise

    key_map, value_map, block_map = {}, {}, {}
    confidence_map, flagged_fields, tables, confidence_scores = {}, [], [], []

    # Build block maps
    for block in response["Blocks"]:
        block_id = block["Id"]
        block_map[block_id] = block
        if block["BlockType"] == "KEY_VALUE_SET":
            if "KEY" in block.get("EntityTypes", []):
                key_map[block_id] = block
            else:
                value_map[block_id] = block

    # Process key-value pairs
    field_map = {}
    for key_block in key_map.values():
        key_text, _ = extract_text_blocks(key_block, block_map, True)
        for rel in key_block.get("Relationships", []):
            if rel["Type"] == "VALUE":
                for value_id in rel["Ids"]:
                    value_block = value_map.get(value_id)
                    value_text, confidence = extract_text_blocks(value_block, block_map, True)
                    needs_review = confidence is not None and confidence < confidence_threshold
                    
                    field_map[key_text] = {
                        "value": value_text,
                        "confidence": confidence,
                        "needs_review": needs_review
                    }
                    confidence_map[key_text] = confidence
                    
                    if confidence is not None:
                        confidence_scores.append(confidence)
                        if needs_review:
                            flagged_fields.append({
                                "field": key_text,
                                "value": value_text,
                                "confidence": confidence,
                                "needs_review": True
                            })

    # Process tables
    for block in response["Blocks"]:
        if block["BlockType"] == "TABLE":
            table = []
            for rel in block.get("Relationships", []):
                if rel["Type"] == "CHILD":
                    for child_id in rel["Ids"]:
                        cell = block_map.get(child_id)
                        if cell and cell["BlockType"] == "CELL":
                            table.append({
                                "row": cell["RowIndex"],
                                "col": cell["ColumnIndex"],
                                "text": extract_text_blocks(cell, block_map, False)[0]
                            })
            tables.append(table)

    # Format as prompt
    prompt = format_extraction_as_prompt(field_map, tables)
    
    image_level_confidence = calculate_average_confidence(confidence_scores)

    return {
        "fields": field_map,
        "tables": tables,
        "prompt": prompt,
        "confidence": confidence_map,
        "flagged": flagged_fields,
        "image_confidence": image_level_confidence,
        "output": field_map
    }


def format_extraction_as_prompt(fields: Dict, tables: List) -> str:
    """Format extracted fields and tables as a prompt for GPT"""
    prompt_lines = ["I have extracted the following fields from the meter slip using OCR:"]
    
    # Format fields
    for key, meta in fields.items():
        value = meta.get("value", "")
        conf = meta.get("confidence")
        needs_review = meta.get("needs_review")
        
        line = f"- {key.strip()}: {value.strip() or '[Missing]'}"
        if conf is not None:
            line += f" (Confidence: {conf}%)"
            if needs_review:
                line += " [NEEDS REVIEW]"
        prompt_lines.append(line)

    # Format tables
    if tables:
        prompt_lines.append("\nAdditionally, here is the tabular data extracted from the slip (used for old/smart meter details):")
        for i, table in enumerate(tables):
            prompt_lines.append(f"\nTable {i + 1}:")
            table.sort(key=lambda x: (x["row"], x["col"]))
            
            # Process rows
            current_row = -1
            row_items = []
            for cell in table:
                if cell["row"] != current_row:
                    if row_items:
                        prompt_lines.append(" | ".join(row_items))
                    row_items = []
                    current_row = cell["row"]
                row_items.append(cell["text"])
            
            # Add the last row
            if row_items:
                prompt_lines.append(" | ".join(row_items))

    prompt_lines.append("\nPlease verify this against the form image and extract the correct values in JSON format as per the schema.")
    return "\n".join(prompt_lines)


# === OpenAI Integration ===
def build_gpt_messages(current_image_base64: str, current_ocr_text: str) -> List[Dict]:
    """Build message sequence for GPT with example and current OCR data"""
    try:
        with open('example_data.json', 'r') as f:
            example_data = json.load(f)
            example_ocr_text = example_data['ocr_text']
            example_json = example_data['json_output']
    except FileNotFoundError:
        logger.warning("Example data file not found, using hardcoded example instead")
        # Fallback to hardcoded examples if file not found
        with open('example_data.json', 'r') as f:
            example_data = json.load(f)
            example_ocr_text = example_data['ocr_text']
            example_json = example_data['json_output']

    example_image_base64 = encode_image(CONFIG["example_image_path"])[1]
    
    system_message = """You are a highly skilled OCR assistant for extracting structured data from scanned meter replacement slips.

                        Your task is to interpret both OCR output and image layout to extract accurate JSON fields.

                        ---

                        Special Handling Instructions:

                        1. **Consumer Account Number**:
                            - Always 12 digits long
                           
                        2. **Meter Number (Old and Smart)**:
                            - Format: `2–3 uppercase letters` followed by `5–6 digits`
                            - Examples:
                                - `GE887158`
                                - `AIK248431`
                            -  Sometimes,  initial characters are not written 
                        3.  Common OCR misreads:
                                - 0 ↔ 8
                                - 1 ↔ 7
                                - 5 ↔ 6
                                - 9 ↔ 8

                        ---

                       Never guess or invent field values. Leave them blank if OCR or image is unclear.

                       Return data in structured JSON format as defined in the schema.
                        """
    
    return [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Example OCR Output:\n{example_ocr_text}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{example_image_base64}"}},
                {"type": "text", "text": f"Expected JSON Output:\n{json.dumps(example_json, indent=2)}"}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"OCR Output:\n{current_ocr_text}"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{current_image_base64}"}}
            ]
        }
    ]


def create_schema_from_default() -> Dict:
    """Generate a schema definition from the default output structure"""
    def convert_to_schema(obj: Dict) -> Dict:
        schema_obj = {"type": "object", "properties": {}}
        for key, value in obj.items():
            if isinstance(value, dict):
                schema_obj["properties"][key] = convert_to_schema(value)
            else:
                schema_obj["properties"][key] = {"type": "string"}
        return schema_obj

    base_schema = convert_to_schema(DEFAULT_OUTPUT)
    
    return {
        "name": "extract_form_data",
        "description": "Extract structured fields from scanned handwritten meter replacement slips",
        "parameters": base_schema
    }


# def call_openai(image_base64: str, ocr_text: str) -> Dict:
#     """Call OpenAI API to extract structured data from OCR text and image"""
#     try:
#         # Get API key from environment variable
#         api_key = os.getenv("OPENAI_API_KEY", "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")
#         openai.api_key = api_key
        
#         messages = build_gpt_messages(image_base64, ocr_text)
        
#         # Use function calling with schema
#         schema = create_schema_from_default()
        
#         response = openai.ChatCompletion.create(
#             model="gpt-4o",
#             messages=messages,
#             tools=[{"type": "function", "function": schema}],
#             tool_choice={"type": "function", "function": {"name": "extract_form_data"}}
#         )
        
#         tool_calls = response["choices"][0]["message"].get("tool_calls", [])
#         if tool_calls:
#             arguments = tool_calls[0]["function"]["arguments"]
#             return json.loads(arguments)
#         else:
#             logger.warning("No structured tool output received from OpenAI")
#             return {"error": "No structured tool output"}
    
#     except Exception as e:
#         logger.error(f"OpenAI API error: {str(e)}")
#         return {"error": f"API error: {str(e)}"}



import time
import openai.error

# def call_openai(image_base64: str, ocr_text: str) -> Dict:
#     """Call OpenAI API to extract structured data from OCR text and image with retry on rate limit"""
#     max_retries = 5
#     base_delay = 5  # seconds

#     for attempt in range(1, max_retries + 1):
#         try:
#             # Set API key
#             api_key = os.getenv("OPENAI_API_KEY", "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")
#             openai.api_key = api_key
        
#             messages = build_gpt_messages(image_base64, ocr_text)
#             schema = create_schema_from_default()

#             response = openai.ChatCompletion.create(
#                 model="gpt-4o",
#                 messages=messages,
#                 tools=[{"type": "function", "function": schema}],
#                 tool_choice={"type": "function", "function": {"name": "extract_form_data"}}
#             )

#             tool_calls = response["choices"][0]["message"].get("tool_calls", [])
#             if tool_calls:
#                 arguments = tool_calls[0]["function"]["arguments"]
#                 return json.loads(arguments)
#             else:
#                 logger.warning("No structured tool output received from OpenAI")
#                 return {"error": "No structured tool output"}

#         except openai.error.RateLimitError as e:
#             wait_time = base_delay * attempt
#             logger.warning(f"Rate limit reached. Retrying in {wait_time} seconds (attempt {attempt}/{max_retries})...")
#             time.sleep(wait_time)

#         except Exception as e:
#             logger.error(f"OpenAI API error: {str(e)}")
#             return {"error": f"API error: {str(e)}"}

#     logger.error("Max retries exceeded for OpenAI API.")
#     return {"error": "Rate limit error: Max retries exceeded"}

def call_openai(image_base64: str, ocr_text: str) -> Dict:
    """Call OpenAI API to extract structured data from OCR text and image with retry on rate limit."""
    max_retries = 5
    base_delay = 5  # seconds

    api_key = os.getenv("OPENAI_API_KEY", "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")
    openai.api_key = api_key

    messages = build_gpt_messages(image_base64, ocr_text)
    schema = create_schema_from_default()

    for attempt in range(1, max_retries + 1):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                tools=[{"type": "function", "function": schema}],
                tool_choice={"type": "function", "function": {"name": "extract_form_data"}}
            )

            tool_calls = response["choices"][0]["message"].get("tool_calls", [])
            if tool_calls:
                arguments = tool_calls[0]["function"]["arguments"]
                return json.loads(arguments)
            else:
                logger.warning("No structured tool output received from OpenAI.")
                return {"error": "No structured tool output"}

        except openai.error.RateLimitError as e:
            # If OpenAI provides a recommended retry-after time, use it
            retry_after = getattr(e, 'retry_after', None)
            if retry_after is None:
                wait_time = base_delay * attempt  # Exponential backoff
            else:
                wait_time = float(retry_after) + 1.0  # Add a small buffer

            logger.warning(f"Rate limit reached. Retrying in {wait_time:.1f} seconds (attempt {attempt}/{max_retries})...")
            time.sleep(wait_time)

        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return {"error": f"API error: {str(e)}"}

    logger.error("Max retries exceeded for OpenAI API.")
    return {"error": "Rate limit error: Max retries exceeded"}



def merge_with_defaults(output: Dict, defaults: Dict) -> Dict:
    """Recursively merge output with default structure"""
    result = deepcopy(defaults)
    for key, value in output.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_with_defaults(value, result[key])
        else:
            result[key] = value
    return result


def process_image_file(image_path: str) -> Dict:
    """Process a single image file through the entire pipeline"""
    try:
        logger.info(f"Processing image: {image_path}")
        
        # Step 1: Load and encode image
        image_bytes, image_base64 = encode_image(image_path)
        
        # Step 2: Extract text using Textract
        textract_result = extract_text_with_textract(image_bytes)
        
        # Step 3: Use GPT to extract structured data
        structured_output = call_openai(image_base64, textract_result["prompt"])
        
        # Step 4: Merge with defaults and add metadata
        final_output = merge_with_defaults(structured_output, DEFAULT_OUTPUT)
        final_output["overall_confidence"] = textract_result.get("image_confidence", 0.0)
        final_output["image_file_name"] = os.path.basename(image_path)
        
        return final_output
    
    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        error_output = deepcopy(DEFAULT_OUTPUT)
        error_output["error"] = str(e)
        error_output["image_file_name"] = os.path.basename(image_path)
        return error_output


def process_image_batch(image_folder: str) -> List[Dict]:
    """Process a batch of images in parallel"""
    image_paths = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.lower().endswith(CONFIG["supported_extensions"])
    ]
    
    if not image_paths:
        logger.warning(f"No images found in {image_folder}")
        return []
    
    logger.info(f"Starting parallel processing of {len(image_paths)} images...")
    output_data = []
    
    with ThreadPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        futures = {executor.submit(process_image_file, path): path for path in image_paths}
        for future in as_completed(futures):
            try:
                result = future.result()
                output_data.append(result)
                logger.info(f"Processed: {result.get('image_file_name', 'unknown')}")
            except Exception as e:
                path = futures[future]
                logger.error(f"Failed to process {path}: {str(e)}")
    
    return output_data


def main():
    """Main entry point for the application"""
    try:
        output_data = process_image_batch(CONFIG["input_folder"])
        
        # Save results to file
        with open(CONFIG["output_file"], "w") as outfile:
            json.dump(output_data, outfile, indent=2)
        
        logger.info(f"Batch processing complete. Output saved to '{CONFIG['output_file']}'.")
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")


if __name__ == "__main__":
    main()