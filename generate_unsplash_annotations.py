import os
import glob
import json
import random
import argparse
import time
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import google.generativeai as genai

# --- Gemini API 调用核心函数 (带详细日志) ---

def annotate_image_with_gemini(image_path: Path, model: genai.GenerativeModel) -> list[dict] | None:
    """
    Uses the Gemini Pro Vision model to identify interesting regions in an image.
    Includes detailed logging for each step.
    """
    print(f"\n--- Processing Image: {image_path.name} ---")
    try:
        # 1. 尝试打开图片
        print(f"   [1/5] Opening image file...")
        image = Image.open(image_path)
        
        # 2. 验证图片完整性
        print(f"   [2/5] Verifying image integrity...")
        image.verify()
        
        # 3. 重新打开图片（verify()后需要）
        print(f"   [3/5] Re-opening image post-verification...")
        image = Image.open(image_path)
        
        # 4. 准备发送 API 请求
        prompt = """
        Analyze this high-resolution image. Your task is to act as an expert computer vision annotator.
        Identify up to 3 small but interesting regions that would require zooming in to see the details clearly.
        For each identified region, provide:
        1.  A tight bounding box in `[x_min, y_min, x_max, y_max]` format. The coordinates must be absolute pixel values.
        2.  A short, descriptive caption of what is inside that region.

        Return the output ONLY as a valid JSON list of objects. Do not include any other text, explanations, or markdown formatting like ```json ... ```.
        If you cannot find any interesting small regions, return an empty list [].

        Example of a perfect response:
        [
          {"box": [1024, 850, 1080, 910], "desc": "A tiny hummingbird sipping nectar from a flower."},
          {"box": [250, 300, 270, 325], "desc": "The brand name 'Nikon' written on the camera strap."}
        ]
        """
        print(f"   [4/5] Sending request to Gemini API (this may take a few seconds)...")
        response = model.generate_content([prompt, image])
        print(f"   [5/5] Received response from Gemini. Parsing...")
        
        # 5. 解析和验证响应
        cleaned_response_text = response.text.strip().replace("```json", "").replace("```", "").strip()

        if not cleaned_response_text:
            print("   -> Result: Gemini returned an empty response. Skipping.")
            return []
            
        regions = json.loads(cleaned_response_text)
        
        if isinstance(regions, list):
            # 过滤掉任何格式不正确的条目
            valid_regions = [
                r for r in regions 
                if isinstance(r, dict) and 'box' in r and 'desc' in r
            ]
            print(f"   -> Result: Successfully parsed {len(valid_regions)} valid regions.")
            return valid_regions
        else:
            print(f"   -> WARNING: Gemini returned a valid JSON, but it's not a list. Type: {type(regions)}. Skipping.")
            return []

    except UnidentifiedImageError:
        print(f"   -> ERROR: Pillow could not identify image file. It may be corrupted: {image_path.name}. Skipping.")
        return None
    except json.JSONDecodeError:
        print(f"   -> ERROR: Failed to parse JSON from Gemini response for {image_path.name}.")
        print(f"      Response snippet: '{response.text[:150]}...' Skipping.")
        return None
    except Exception as e:
        print(f"   -> ERROR: An unexpected error occurred for {image_path.name}: {e}. Skipping.")
        return None

# --- 主执行逻辑 (带详细日志) ---

def main(args):
    """
    Main function to orchestrate the annotation generation process.
    """
    print("--- Step 1: Configuring Gemini API ---")
    try:
        api_key = os.environ.get(args.api_key_var)
        if not api_key:
            raise ValueError(f"API key environment variable '{args.api_key_var}' not found. Please set it.")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(args.model_name)
        print("   -> Gemini API configured successfully.")
    except Exception as e:
        print(f"   -> FATAL ERROR: Could not configure Gemini API: {e}")
        return

    print("\n--- Step 2: Setting up State and Checking for Resumption ---")
    all_annotations = []
    processed_files = set()
    output_path = Path(args.output_file)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and os.path.getsize(output_path) > 0:
        print(f"   -> Found existing output file: {output_path}. Attempting to resume.")
        with open(output_path, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    if line.strip():
                        annotation = json.loads(line)
                        all_annotations.append(annotation)
                        processed_files.add(Path(annotation['source_image']).name)
                except json.JSONDecodeError:
                    print(f"   -> Warning: Malformed JSON on line {line_num+1} in existing file. Skipping line.")
        print(f"   -> Resumed with {len(all_annotations)} existing annotations from {len(processed_files)} unique files.")
    else:
        print("   -> No existing data found. Starting a new session.")

    print("\n--- Step 3: Building and Shuffling Image Pool ---")
    print(f"   -> Scanning image directory: {args.image_dir}...")
    try:
        all_image_paths = [Path(p) for p in glob.glob(os.path.join(args.image_dir, "*.jpg"))]
        print(f"   -> Found {len(all_image_paths)} total images.")
    except Exception as e:
        print(f"   -> FATAL ERROR: Could not read image directory: {e}")
        return

    unprocessed_paths = [p for p in all_image_paths if p.name not in processed_files]
    print(f"   -> {len(unprocessed_paths)} images remain to be processed.")
    
    if not unprocessed_paths:
        print("   -> All available images have already been processed.")
        return

    pool_size = min(len(unprocessed_paths), args.pool_size)
    image_pool = random.sample(unprocessed_paths, pool_size)
    print(f"   -> Created a random pool of {len(image_pool)} images for this run.")

    print("\n--- Step 4: Starting Main Annotation Loop ---")
    pbar = tqdm(total=args.target_count, initial=len(all_annotations), desc="Total Annotations")
    
    image_counter = 0
    for image_path in image_pool:
        if len(all_annotations) >= args.target_count:
            print(f"\nTarget of {args.target_count} annotations reached. Stopping.")
            break
        
        image_counter += 1
        print(f"\n[Processing Image {image_counter}/{len(image_pool)} in current pool | Total Annotations: {len(all_annotations)}/{args.target_count}]")
        
        regions = annotate_image_with_gemini(image_path, model)
        processed_files.add(image_path.name) # 标记为已处理，无论成功与否
        
        if regions:
            for region in regions:
                annotation = {
                    "source_image": str(image_path.resolve()), # 使用绝对路径
                    "bounding_box": region['box'],
                    "description": region['desc']
                }
                all_annotations.append(annotation)
                pbar.update(1)

        # 定期保存
        if image_counter % args.save_interval == 0:
            print(f"\n[Saving progress... ({len(all_annotations)} annotations so far)]")
            with open(output_path, 'w') as f:
                for annotation in all_annotations:
                    f.write(json.dumps(annotation) + '\n')
        
        print(f"   -> Sleeping for {args.sleep_interval} second(s)...")
        time.sleep(args.sleep_interval)

    print("\n--- Step 5: Finalizing and Saving ---")
    with open(output_path, 'w') as f:
        for annotation in all_annotations:
            f.write(json.dumps(annotation) + '\n')
            
    pbar.close()
    print(f"\nProcess complete. A total of {len(all_annotations)} annotations were generated.")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ZOOM-IN annotations for Unsplash images using the Gemini API.")
    # ... (命令行参数部分与之前完全相同)
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing the Unsplash JPEG images.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file where annotations will be saved.")
    parser.add_argument("--target_count", type=int, default=500, help="The target number of annotations to generate before stopping.")
    parser.add_argument("--pool_size", type=int, default=2000, help="The number of random images to select from the source directory to form the processing pool.")
    parser.add_argument("--model_name", type=str, default="gemini-1.5-pro-latest", help="The Gemini model to use for annotation.")
    parser.add_argument("--api_key_var", type=str, default="GEMINI_API_KEY", help="The name of the environment variable holding the Gemini API key.")
    parser.add_argument("--save_interval", type=int, default=10, help="How often (in terms of IMAGES PROCESSED) to save the progress.")
    parser.add_argument("--sleep_interval", type=float, default=1.0, help="Seconds to wait between API calls to respect rate limits.")

    args = parser.parse_args()
    main(args)