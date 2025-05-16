#!pip install openai==0.28

import cv2
import numpy as np
from PIL import Image
import io
import openai
import base64


# === Step 1: Enhance the image ===
def enhance_image(image_path):
    # Read and convert to grayscale
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to improve clarity
    scale_percent = 150  # Increase size by 150%
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(gray, dim, interpolation=cv2.INTER_LINEAR)

    # Increase contrast and apply adaptive threshold
    contrast = cv2.convertScaleAbs(resized, alpha=1.5, beta=0)
    thresh = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    # Convert to PIL Image
    pil_img = Image.fromarray(thresh)
    return pil_img


# === Step 2: Convert processed image to base64 ===
def image_to_base64(pil_img):
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")



# Load and encode the image
image_path = "/content/WhatsApp Image 2025-04-22 at 11.46.35 PM.jpeg"
with open(image_path, "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

processed_img = enhance_image(image_path)
#processed_img = preprocess_image(image_path)
image_base64 = image_to_base64(processed_img)


# Prompt
prompt = """
You are an intelligent document parser. Extract all fields from this smart meter slip and return them in this JSON structure:

{
  "consumer_name": "",
  "consumer_address": "",
  "consumer_account_no": "",
  "consumer_mobile_no": "",
  "date_of_installation": "",
  "smart_meter_replacement_slip_no": "",
  "old_meter": {
    "meter_no": "",
    "meter_reading": "",
    "manufacturer": "",
    "md_kw": "",
    "status": ""
  },
  "smart_meter": {
    "meter_no": "",
    "manufacturer": "",
    "status": ""
  },
  "feeder_name": "",
  "subdivision": "",
  "book_no": "",
  "sl_no": "",
  "dtr": "",
  "consumer_signature": "",
  "agency_supervisor_name": "",
  "agency_supervisor_mobile": ""
}

If any field is not visible, leave it as an empty string.
Return only valid JSON.
"""

# OpenAI API key
openai.api_key = "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"  # Replace with your actual key

# API call
try:
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts structured data from scanned documents."},
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
        max_tokens=1000
    )

    print("===== Extracted JSON =====")
    print(response["choices"][0]["message"]["content"])

except Exception as e:
    print("❌ Error:", str(e))
