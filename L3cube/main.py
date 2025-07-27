from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import pandas as pd

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

HUGGING_FACE_TOKEN = ""
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HUGGING_FACE_TOKEN)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    token=HUGGING_FACE_TOKEN
)

model.eval()

def generate_response(prompt_text):
    messages = [
        {"role": "user", "content": prompt_text}
    ]

    # Step 1: Apply the chat template to get the formatted string
    formatted_chat_string = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Step 2: Tokenize the formatted string to get input_ids and attention_mask
    encoded_input = tokenizer(
        formatted_chat_string,
        return_tensors="pt",
        return_attention_mask=True,
        padding=True,
        truncation=True
    )

    input_ids = encoded_input["input_ids"].to(model.device)
    attention_mask = encoded_input["attention_mask"].to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=150,
            num_return_sequences=1,
            pad_token_id=tokenizer.pad_token_id,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )

    # Key Change: Decode without skipping special tokens initially
    # This ensures that tokens like <|start_header_id|> and <|end_header_id|> are preserved
    # which are part of the chat template structure.
    generated_text = tokenizer.decode(output[0], skip_special_tokens=False)

    # Identify the full assistant header as it would appear in the template,
    # ensuring it includes the special tokens if they are part of the template.
    # We apply the chat template with an empty assistant content to get the exact header string.
    assistant_full_header_template = tokenizer.apply_chat_template(
        [{"role": "assistant", "content": ""}],
        tokenize=False,
        add_generation_prompt=False # Do not add system/user if we just want assistant header
    ).strip() # Strip to remove any trailing newlines if present in template itself

    # Find the position of the assistant's turn in the generated text
    # We look for the last occurrence in case the model generates multiple assistant turns or repeats it.
    response_start_index = generated_text.rfind(assistant_full_header_template)

    if response_start_index != -1:
        # Extract the content *after* the assistant's header
        extracted_response = generated_text[response_start_index + len(assistant_full_header_template):].strip()
    else:
        # Fallback if the full structured header is not found.
        # This can happen if the model generates a malformed header or if skip_special_tokens=True was used before.
        # In this fallback, we'll try to find "assistant" (plain text) and strip common prefixes.
        extracted_response = generated_text
        
        # Remove common template parts (plain text versions)
        extracted_response = extracted_response.replace("system\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2025\n\n", "").strip()
        extracted_response = extracted_response.replace("user\n\n", "").strip()
        extracted_response = extracted_response.replace(prompt_text, "").strip() # Remove the original prompt
        
        # Find the last "assistant" plain text marker
        plain_assistant_marker_index = extracted_response.rfind("assistant")
        if plain_assistant_marker_index != -1:
            extracted_response = extracted_response[plain_assistant_marker_index + len("assistant"):].strip()
        
    # Final cleanup: Remove any remaining special tokens, BOS/EOS tokens, or end-of-turn IDs
    # by re-encoding the extracted response and decoding it with skip_special_tokens=True.
    # This ensures only the actual generated text remains.
    final_cleaned_response = tokenizer.decode(
        tokenizer.encode(extracted_response, add_special_tokens=False), # Encode without adding new special tokens
        skip_special_tokens=True # Decode, explicitly skipping all special tokens this time
    ).strip()

    return final_cleaned_response

# --- Function to load prompts from CSV ---
def load_prompts_from_csv(file_path, column_name):
    try:
        df = pd.read_csv(file_path)
        if column_name in df.columns:
            df = df.dropna(subset=[column_name])
            return df[column_name].tolist()
        else:
            print(f"Error: Column '{column_name}' not found in '{file_path}'.")
            return []
    except FileNotFoundError:
        print(f"Error: CSV file not found at '{file_path}'.")
        return []
    except Exception as e:
        print(f"An error occurred while reading '{file_path}': {e}")
        return []

# --- Helper function to append to CSV ---
def append_to_csv(file_path, data_row, header_written):
    df_row = pd.DataFrame([data_row])
    df_row.to_csv(file_path, mode='a', header=not header_written, index=False, encoding='utf-8')


# --- Hindi Inference ---
hindi_csv_path = "Hindi.csv"
hindi_question_column = "Question"
hindi_output_csv = "Hindi_Responses_Cleaned.csv" # New name for the output file
hindi_header_written = False

print("--- Hindi Inference ---")
hindi_prompts = load_prompts_from_csv(hindi_csv_path, hindi_question_column)

if hindi_prompts:
    # Remove the output file if it exists to ensure a clean start
    if os.path.exists(hindi_output_csv):
        os.remove(hindi_output_csv)
        print(f"Removed existing file: {hindi_output_csv}")

    for i, prompt in enumerate(hindi_prompts):
        print(f"Hindi Prompt {i+1}: {prompt}")
        response = generate_response(prompt)
        print(f"Hindi Response {i+1}: {response}\n")

        append_to_csv(hindi_output_csv, {"Prompt": prompt, "Response": response}, hindi_header_written)
        if not hindi_header_written:
            hindi_header_written = True

    print(f"Finished processing Hindi prompts. Responses are in {hindi_output_csv}")
else:
    print("No Hindi prompts to process.")

# --- Marathi Inference ---
marathi_csv_path = "Marathi.csv"
marathi_question_column = "Question"
marathi_output_csv = "Marathi_Responses_Cleaned.csv" # New name for the output file
marathi_header_written = False

print("\n--- Marathi Inference ---")
marathi_prompts = load_prompts_from_csv(marathi_csv_path, marathi_question_column)

if marathi_prompts:
    # Remove the output file if it exists to ensure a clean start
    if os.path.exists(marathi_output_csv):
        os.remove(marathi_output_csv)
        print(f"Removed existing file: {marathi_output_csv}")

    for i, prompt in enumerate(marathi_prompts):
        print(f"Marathi Prompt {i+1}: {prompt}")
        response = generate_response(prompt)
        print(f"Marathi Response {i+1}: {response}\n")

        append_to_csv(marathi_output_csv, {"Prompt": prompt, "Response": response}, marathi_header_written)
        if not marathi_header_written:
            marathi_header_written = True

    print(f"Finished processing Marathi prompts. Responses are in {marathi_output_csv}")
else:
    print("No Marathi prompts to process.")