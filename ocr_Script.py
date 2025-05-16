import boto3
import base64
import sys


def extract_text_with_textract(image_path):
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()

    client = boto3.client("textract", region_name="us-east-1")

    response = client.analyze_document(
        Document={"Bytes": image_bytes},
        FeatureTypes=["FORMS", "TABLES"]
    )

    key_map = {}
    value_map = {}
    block_map = {}
    tables = []

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

    for block in response['Blocks']:
        if block['BlockType'] == 'TABLE':
            table = []
            for rel in block.get('Relationships', []):
                if rel['Type'] == 'CHILD':
                    for child_id in rel['Ids']:
                        cell = block_map.get(child_id)
                        if cell and cell['BlockType'] == 'CELL':
                            row_idx = cell['RowIndex']
                            col_idx = cell['ColumnIndex']
                            cell_text = get_text(cell, block_map)
                            table.append({"row": row_idx, "col": col_idx, "text": cell_text})
            tables.append(table)

    def format_field_map_as_prompt(textract_result):
        fields = textract_result.get("fields", {})
        tables = textract_result.get("tables", [])

        prompt_lines = ["I have extracted the following fields from the meter slip using OCR:"]
        for key, value in fields.items():
            prompt_lines.append(f"- {key.strip()}: {value.strip() or '[Missing]'}")

        if tables:
            prompt_lines.append("\nAdditionally, here is the tabular data extracted from the slip (used for old/smart meter details):")
            for i, table in enumerate(tables):
                prompt_lines.append(f"\nTable {i + 1}:")
                table.sort(key=lambda x: (x['row'], x['col']))
                current_row = -1
                row_items = []
                for cell in table:
                    if cell['row'] != current_row:
                        if row_items:
                            prompt_lines.append(" | ".join(row_items))
                        row_items = []
                        current_row = cell['row']
                    row_items.append(cell['text'])
                if row_items:
                    prompt_lines.append(" | ".join(row_items))

        return "\n".join(prompt_lines)

    textract_result = {
        "fields": field_map,
        "tables": tables
    }
    #return format_field_map_as_prompt(textract_result)
    return textract_result

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python textract_ocr_to_prompt.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    prompt = extract_text_with_textract(image_path)
    print("\n=== OCR Prompt Output ===\n")
    print(prompt)
