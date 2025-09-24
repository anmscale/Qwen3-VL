import json
import os
from typing import Dict, List


def load_coco_annotations(annotation_file: str) -> Dict:
    with open(annotation_file, "r") as f:
        return json.load(f)


def build_image_id_to_file_name(coco: Dict) -> Dict[int, str]:
    return {img["id"]: img["file_name"] for img in coco.get("images", [])}


def build_annotations(coco: Dict, image_folder_rel: str, max_captions_per_image: int = 5) -> List[Dict]:
    image_id_to_file = build_image_id_to_file_name(coco)
    image_id_to_captions: Dict[int, List[str]] = {}
    for ann in coco.get("annotations", []):
        img_id = ann.get("image_id")
        caption = ann.get("caption", "").strip()
        if not caption:
            continue
        image_id_to_captions.setdefault(img_id, []).append(caption)

    records: List[Dict] = []
    for img_id, file_name in image_id_to_file.items():
        captions = image_id_to_captions.get(img_id, [])
        if not captions:
            continue
        # Cap the number of captions to keep dataset size manageable if desired
        if max_captions_per_image > 0:
            captions = captions[:max_captions_per_image]

        for cap in captions:
            records.append(
                {
                    "image": os.path.join(image_folder_rel, file_name),
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\nDescribe this image.",
                        },
                        {
                            "from": "gpt",
                            "value": cap,
                        },
                    ],
                }
            )

    return records


def convert(
    coco_annotation_path: str,
    image_folder_rel: str,
    output_path: str,
    max_captions_per_image: int = 5,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    coco = load_coco_annotations(coco_annotation_path)
    data = build_annotations(coco, image_folder_rel, max_captions_per_image)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {len(data)} records to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert COCO captions to QwenVL format")
    parser.add_argument("--coco_annotation", type=str, required=True, help="Path to COCO captions JSON, e.g., captions_train2017.json")
    parser.add_argument(
        "--image_folder_rel",
        type=str,
        required=True,
        help="Image folder path relative to data_path stored in dataset registry, e.g., train2017",
    )
    parser.add_argument("--output", type=str, required=True, help="Output JSON path for QwenVL annotations")
    parser.add_argument(
        "--max_captions_per_image",
        type=int,
        default=5,
        help="Limit the number of captions per image (<=5 in COCO captions)",
    )

    args = parser.parse_args()
    convert(
        coco_annotation_path=args.coco_annotation,
        image_folder_rel=args.image_folder_rel,
        output_path=args.output,
        max_captions_per_image=args.max_captions_per_image,
    )



