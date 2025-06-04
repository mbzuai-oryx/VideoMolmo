import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import re
from PIL import Image
import os

SYSTEM_PROMPT = "You are given {num_frames} frames from a video. The frame indices are {selected_frame_idxs}. "
PROMPT_TEMPLATES = [
        "Point to {label}\nPlease say 'This isn't in the video.' if it is not in the video.",
        "Point to all occurrences of \"{label}\"",
        "Point to any {label} in the video",
        "Point to any {label} in the video.",
        "Point: Where are the {label}",
        "Show me where the {label} are",
        "Can you show me where the {label} are?",
        "Show me where the {label} are",
        "Show me where a {label} is",
        "Show me where a {label} is.",
        "If there are any {label} in the video? Show me where they are.",
        "Where are the {label}?",
        "Generate a list of points showing where the {label} are.",
        "Find the \"{label}\".",
        "Find a \"{label}\".",
        "Locate all {label}.",
        "Locate an {label}.",
        "Locate a {label}.",
        "Locate every {label}.",
        "Locate {label}.",
        "Locate the {label}.",
        "Object: {label}\nInstruction: Point to the object.",
        "find {label}",
        "find {label}.",
        "Point to every {label}",
        "find any {label} in the picture",
        "Find the {label}",
        "Find any {label}",
        "Point to a {label}",
        "Point to an {label}",
        "Look for {label} in the video and show me where they are.",
        "Help me find an object in the video by pointing to them.\nObject: {label}.",
        "I am looking for {label}, where can they be found in the video?",
        "Can you see any {label} in the video? Point to them.",
        "Point out each {label} in the video.",
        "Point out every {label} in the video.",
        "Point to the {label} in the video.",
        "Locate each {label} in the video.",
        "Can you point out all {label} in this video?",
        "Please find {label} and show me where they are.",
        "If there are any {label} present, indicate their positions.",
        "If there is a {label} present, indicate its positions.",
        "show me all visible {label}",
    ]

def add_points(predictor, input_points, input_labels, inference_state, f=0, device='cuda'):
    for i in range(len(input_points)):
        input_point = np.array([input_points[i]])
        input_label = np.array([input_labels[i]])
        ann_frame_idx = f # Frame index to interact/start with.
        ann_object_id = i # Give a unique object ID to the object, an integer.

        with torch.inference_mode(), torch.autocast(str(device), dtype=torch.bfloat16):
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=ann_object_id,
                points=input_point,
                labels=input_label
            )
    return out_obj_ids, out_mask_logits

def draw_point_and_show(image_path=None, points=None):
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    for point in points:
        image = cv2.circle(
            image, 
            (point[0], point[1]), 
            radius=5, 
            color=(0, 255, 0), 
            thickness=5,
            lineType=cv2.LINE_AA
        )

    plt.imshow(image[..., ::-1])
    plt.axis('off')
    plt.show()


def get_coords(output_string, image):
    w, h = image.size
    coordinates = []
    if '<points' in output_string:
        # Handle multiple coordinates
        matches = re.findall(r'(x\d+)="([\d.]+)" (y\d+)="([\d.]+)"', output_string)
        coordinates = [(int(float(x_val)/100*w), int(float(y_val)/100*h)) for _, x_val, _, y_val in matches]
    else:
        # Handle single coordinate
        match = re.search(r'x="([\d.]+)" y="([\d.]+)"', output_string)
        if match:
            coordinates = [(int(float(match.group(1))/100*w), int(float(match.group(2))/100*h))]    
    return coordinates


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap('tab10')
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    ax.axis('off')

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def load_png_as_array(directory):
    boolean_arrays = []
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.png'):
            file_path = os.path.join(directory, filename)
            # Open the image and convert to grayscale
            image = Image.open(file_path).convert('L')
            # Convert image to a NumPy array
            image_array = np.array(image)
            # Convert to a boolean array (nonzero pixels -> True, zero pixels -> False)
            boolean_array = image_array > 0
            boolean_arrays.append(boolean_array)
    return boolean_arrays

# Function to recursively convert ndarrays to lists
def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_ndarray(i) for i in obj]
    return obj


import shutil
from natsort import natsorted  # Ensures correct sorting of filenames

def save_reversed_frames(input_dir, output_dir):
    # Get all frame filenames and sort them naturally (e.g., frame_1.png, frame_2.png, ...)
    frame_files = natsorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not frame_files:
        print("Error: No image frames found in the directory.")
        return
    
    # Reverse the order of frames
    frame_files.reverse()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save reversed frames by copying them to the output directory
    for idx, frame_file in enumerate(frame_files):
        src_path = os.path.join(input_dir, frame_file)
        # Preserve the original file extension
        _, ext = os.path.splitext(frame_file)
        dest_filename = f"{idx:05d}{ext}"
        dest_path = os.path.join(output_dir, dest_filename)
        shutil.copy2(src_path, dest_path)

    print("Reversed frames saved successfully!")

def save_images_folder(np_images, video_name):
    # Save each image in the list as a JPEGx
    if not os.path.exists(video_name):
        os.makedirs(video_name)
    for idx, img in enumerate(np_images):
        save_path = os.path.join(video_name, f"{idx}.jpg")
        img_pil = Image.fromarray(img.astype(np.uint8))  # Convert NumPy array to PIL Image
        img_pil.save(save_path, "JPEG")

def pil_to_np(images):
    image_arrays = []
    for image in images:
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            image_arrays.append(np.array(image))
    images = image_arrays
    return images


import re

# placeholder nouns by Wh-word
_PLACEHOLDERS = {
    "who":      "person",
    "what":     "object",
    "which":    "object",
    "where":    "location",
    "when":     "time",
    "why":      "reason",
    "how":      "manner",
}

# strip leading Wh-word + auxiliaries (is/are/does/can/…)
_leading_pattern = re.compile(
    r"^(?P<wh>which|what|where|who|when|why|how)\b"         
    r"(?:\s+(?:is|are|was|were|do|does|did|can|could|should|"
    r"would|has|have|had|will|may|might|must))*\s+",           
    flags=re.IGNORECASE
)

def _to_gerund(verb: str) -> str:
    """
    Very simple heuristic to turn a 3rd-person singular verb like 'rides' -> 'riding'.
    Handles common -s -> -ing, including doubling consonant e.g. 'run'->'running'.
    """
    verb = verb.lower()
    if verb.endswith("ies"):
        return verb[:-3] + "ying"
    if verb.endswith("es") and len(verb) > 3:
        return verb[:-2] + "ing"
    if verb.endswith("s") and len(verb) > 2:
        stem = verb[:-1]
        # double final consonant when CVC
        if re.match(r".*[aeiou][bcdfghjklmnpqrstvwxyz]$", stem):
            return stem + stem[-1] + "ing"
        return stem + "ing"
    return verb + "ing"


def get_pointing_prompt(question: str) -> str:
    """
    Convert a Wh-question into a "Point to ..." prompt, including
    special handling for 'what/which does' structures.
    """
    q = question.strip().rstrip("?").strip()
    m = _leading_pattern.match(q)
    if m:
        wh = m.group("wh").lower()
        remainder = q[m.end():].strip()
    else:
        wh = "what"
        remainder = q

    placeholder = _PLACEHOLDERS.get(wh, "object")

    # Special case: 'what/which does <subject> <verb>' -> point to the object being <verb>ed
    if wh in ("what", "which"):
        # split remainder into subject phrase and verb
        parts = remainder.rsplit(" ", 1)
        if len(parts) == 2 and re.match(r"^[a-zA-Z]+$", parts[1]):
            subj, verb = parts
            gerund = _to_gerund(verb)
            # ensure subject has article
            if not re.match(r"^(the|a|an)\b", subj, re.IGNORECASE):
                subj = f"the {subj}"
            remainder = f"{placeholder} {subj} is {gerund}"

    # For 'who' with finite verb at start, convert to gerund
    if wh == "who":
        tokens = remainder.split()
        if tokens:
            tokens[0] = _to_gerund(tokens[0])
            remainder = " ".join(tokens)

    # if remainder starts with verb/prep, prefix placeholder noun
    if re.match(r"^(on|in|at|under|over|above|below|next|looking|standing|sitting|lying|riding)\b",
                remainder, re.IGNORECASE):
        remainder = f"{placeholder} {remainder}"

    # ensure an article
    if not re.match(r"^(the|a|an)\b", remainder, re.IGNORECASE):
        remainder = f"the {remainder}"

    prompt = f"Point to {remainder}."
    return prompt[0].upper() + prompt[1:]
