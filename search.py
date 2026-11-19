from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"
PHOTO_DIR = Path("photos")
CACHE_FILE = Path("photo_embeddings.pt")
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png"}

print("Loading CLIP model...")
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()

PHOTO_DIR.mkdir(exist_ok=True)
photo_paths = sorted(
    path for path in PHOTO_DIR.iterdir()
    if path.is_file() and path.suffix.lower() in SUPPORTED_FORMATS
)
if not photo_paths:
    raise SystemExit("Add JPG or PNG images to the photos folder.")

# File metadata helps invalidate the cache when the collection changes.
photo_info = [
    (str(path), path.stat().st_mtime_ns, path.stat().st_size)
    for path in photo_paths
]


def encode_image(path):
    with Image.open(path) as image:
        inputs = processor(images=image.convert("RGB"), return_tensors="pt")

    with torch.inference_mode():
        features = model.get_image_features(**inputs).pooler_output

    return torch.nn.functional.normalize(features, dim=-1)


def encode_text(query):
    inputs = processor(text=[query], return_tensors="pt", padding=True)

    with torch.inference_mode():
        features = model.get_text_features(**inputs).pooler_output

    return torch.nn.functional.normalize(features, dim=-1)


def create_image_embeddings():
    embeddings = []
    valid_paths = []

    print("Processing photos...")
    for path in photo_paths:
        try:
            embeddings.append(encode_image(path))
            valid_paths.append(path)
        except Exception as error:
            print(f"Skipping {path.name}: {error}")

    if not embeddings:
        raise SystemExit("No readable photos found.")

    return torch.cat(embeddings), valid_paths


def load_embeddings():
    if CACHE_FILE.exists():
        try:
            cache = torch.load(CACHE_FILE, map_location="cpu", weights_only=True)
            if (
                cache["photo_info"] == photo_info
                and cache["model_name"] == MODEL_NAME
                and len(cache["valid_paths"]) == cache["embeddings"].shape[0]
            ):
                print("Using saved image embeddings.")
                return cache["embeddings"], [
                    Path(path) for path in cache["valid_paths"]
                ]
        except (OSError, KeyError, RuntimeError, ValueError, TypeError):
            pass

    embeddings, paths = create_image_embeddings()
    torch.save(
        {
            "photo_info": photo_info,
            "model_name": MODEL_NAME,
            "embeddings": embeddings,
            "valid_paths": [str(path) for path in paths],
        },
        CACHE_FILE,
    )
    print("Image embeddings saved.")
    return embeddings, paths


def show_results(paths, scores, heading):
    fig, axes = plt.subplots(
        1, len(paths), figsize=(5 * len(paths), 5), squeeze=False
    )
    fig.suptitle(heading, fontsize=16)

    for position, (path, score) in enumerate(zip(paths, scores)):
        ax = axes[0][position]
        with Image.open(path) as image:
            ax.imshow(image.convert("RGB"))

        ax.set_title(
            f"Match {position + 1}\n{path.name}\nSimilarity: {score:.3f}",
            fontsize=10,
        )
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close(fig)


image_embeddings, valid_paths = load_embeddings()


def search_photos(query_embedding, heading, exclude=None):
    scores = (image_embeddings @ query_embedding.T).flatten()

    # Avoid returning the exact reference file as its own closest match.
    if exclude is not None:
        for index, path in enumerate(valid_paths):
            if path.resolve() == exclude.resolve():
                scores[index] = float("-inf")

    available = int(torch.isfinite(scores).sum().item())
    if not available:
        print("No other photos available to compare.")
        return

    top_scores, top_indices = torch.topk(scores, k=min(3, available))
    matching_paths = []

    print("\nTop matching photos:\n")
    for rank, (score, index) in enumerate(zip(top_scores, top_indices), start=1):
        path = valid_paths[index.item()]
        matching_paths.append(path)
        print(f"{rank}. {path.name} | Cosine similarity: {score.item():.3f}")

    show_results(matching_paths, top_scores.tolist(), heading)


while True:
    print("\nAI PHOTO SEARCH")
    print("1. Search by description")
    print("2. Search by image")
    print("3. Exit")

    choice = input("\nChoose an option (1-3): ").strip()

    if choice == "1":
        query = input("\nDescribe the photo you're looking for: ").strip()
        if not query:
            print("Please enter a description.")
            continue

        print("\nSearching your photos...")
        search_photos(encode_text(query), f'Search results for "{query}"')

    elif choice == "2":
        image_path = input("\nEnter the reference image path: ").strip()
        reference = Path(image_path.strip('"').strip("'"))

        if not reference.is_file():
            print("Image not found. Check the file path.")
            continue

        try:
            print("\nFinding similar photos...")
            search_photos(
                encode_image(reference),
                f"Photos similar to {reference.name}",
                exclude=reference,
            )
        except (OSError, ValueError) as error:
            print(f"Could not open the image: {error}")

    elif choice == "3":
        print("Goodbye!")
        break
    else:
        print("Please choose 1, 2, or 3.")
