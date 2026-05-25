from datasets import load_dataset
from openai import OpenAI
import pandas as pd
import os
import time
import io
import base64
import numpy as np
from PIL import Image
from multiprocessing import Process

# ---------------- CONFIG ----------------
DATASET_NAME = "Santhosh2312/rb_llm_test"
MODEL_NAME = "qwen/qwen3-vl-4b"  # e.g. llava, qwen-vl

OUT_PATH = "C:\\Users\\lsant\\Downloads\\rb_llm_lmstudio_qwen.parquet"
SAVE_EVERY = 10
PRINT_EVERY = 10

TARGET_SIZE = (336, 336)

# ---------------- LM STUDIO CLIENT ----------------
client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="sk-dummy"
)

# ---------------- PROMPT ----------------
PROMPT_TEMPLATE = """You are an expert astronomer specializing in transient detection.

You are given:
- Classification Label: {label}
- Detected Features: {features}

Explain why the classification is correct based only on the images.

Output:

**NEW_IMAGE_DESCRIPTION**:
**REFERENCE_IMAGE_DESCRIPTION**:
**DIFFERENCE_IMAGE_DESCRIPTION**:
**CLASSIFICATION RESULT**: {label}
**DETECTION RESULT**: {features}
**REASONING**:

PREDICTION: {label}
"""

# ---------------- HELPERS ----------------

def split_triplet(hf_img):
    if isinstance(hf_img, Image.Image):
        img = hf_img.convert("RGB")
    else:
        img = Image.open(io.BytesIO(hf_img["bytes"])).convert("RGB")

    arr = np.array(img)
    h, w, _ = arr.shape
    w3 = w // 3

    new  = Image.fromarray(arr[:, :w3])
    ref  = Image.fromarray(arr[:, w3:2*w3])
    diff = Image.fromarray(arr[:, 2*w3:])

    new  = new.resize(TARGET_SIZE)
    ref  = ref.resize(TARGET_SIZE)
    diff = diff.resize(TARGET_SIZE)

    return new, ref, diff


def image_to_base64(img):
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def generate_with_retry(prompt, new_b64, ref_b64, diff_b64):
    for i in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{new_b64}"
                                }
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{ref_b64}"
                                }
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{diff_b64}"
                                }
                            }
                        ]
                    }
                ]
                ,
                temperature=0.3,
                max_tokens=1000,
                extra_body={
                                "enable_thinking": False,
                                "thinking_budget": 0
                            }
            )

            return response.choices[0].message.content

        except Exception as e:
            print("Retrying...", e)
            time.sleep(2 ** i)

    return None


def process_chunk(chunk_id, df_chunk, out_path):
    rows = []
    done = 0
    start = time.time()

    for idx, row in df_chunk.iterrows():

        # ---------- LABEL ----------
        labels = list(row["labels"]) if row["labels"] is not None else []

        if len(labels) == 0:
            label = "REAL"
            features = "none"
        else:
            label = "BOGUS"
            features = ", ".join(labels)

        prompt = PROMPT_TEMPLATE.format(label=label, features=features)

        # ---------- IMAGES ----------
        new_img, ref_img, diff_img = split_triplet(row["triplet_image"])

        new_b64 = image_to_base64(new_img)
        ref_b64 = image_to_base64(ref_img)
        diff_b64 = image_to_base64(diff_img)

        # ---------- GENERATION ----------
        output = generate_with_retry(prompt, new_b64, ref_b64, diff_b64)

        if output is None:
            continue

        original_response = row.get("response", "")
        new_response = output

        rows.append({
            "index": idx,
            "instruction": prompt,

            "original_response": original_response,   # from dataset
            "lmstudio_response": new_response,        # new model output

            "labels": labels,
            "features": features,
            "predicted_label": label
        })

        done += 1

        if done % PRINT_EVERY == 0:
            elapsed = time.time() - start
            print(f"[Chunk {chunk_id}] {done}/{len(df_chunk)} | {elapsed/60:.1f} min")

        if len(rows) % SAVE_EVERY == 0:
            pd.DataFrame(rows).to_parquet(out_path)

    pd.DataFrame(rows).to_parquet(out_path)
    print(f"[Chunk {chunk_id}] DONE")


# ---------------- MAIN ----------------

if __name__ == "__main__":

    print("Loading dataset...")
    ds = load_dataset(DATASET_NAME)["train"]

    df = ds.to_pandas()
    print(f"Total samples: {len(df)}")

    CHUNK_SIZE = 50   # adjust based on speed

    all_outputs = []

    for i in range(0, len(df), CHUNK_SIZE):
        chunk = df.iloc[i:i+CHUNK_SIZE].copy()
        out_file = f"chunk_{i//CHUNK_SIZE}.parquet"

        print(f"\nProcessing chunk {i//CHUNK_SIZE} ({i} → {i+len(chunk)})")

        process_chunk(i//CHUNK_SIZE, chunk, out_file)

    # Merge all chunks
    dfs = []
    for i in range((len(df) + CHUNK_SIZE - 1) // CHUNK_SIZE):
        path = f"chunk_{i}.parquet"
        if os.path.exists(path):
            dfs.append(pd.read_parquet(path))

    final = pd.concat(dfs, ignore_index=True)
    final.to_parquet(OUT_PATH)

    print(f"\nDONE → {OUT_PATH}")