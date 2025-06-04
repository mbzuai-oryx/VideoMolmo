import os
import argparse
import json

from PIL import Image
from transformers import (
    GenerationConfig,
    BitsAndBytesConfig,
    AutoTokenizer
)
import numpy as np
import torch
import cv2

from sam2.build_sam import build_sam2_video_predictor
from utils import save_images_folder, get_coords, pil_to_np
from memory_preprocessor import MultiModalPreprocessor
from memory import MolmoForCausalLM  

parser = argparse.ArgumentParser(description="Inference")

parser.add_argument("--dataset", type=str, default='autonomous_driving')
parser.add_argument("--frame_rate", type=int, default=10)
parser.add_argument("--save_path", type=str, default='./results/molmo')
parser.add_argument("--molmo_model", type=str, default='allenai/Molmo-7B-D-0924') 
parser.add_argument("--split", type=int, default=0, choices=[0,1,2,3]) 
parser.add_argument("--sam2_checkpoint", type=str, default='checkpoints/sam2.1_hiera_large.pt')
parser.add_argument("--sam2_config", type=str, default='configs/sam2.1_hiera_l.yaml')
args = parser.parse_args()

device = 'cuda'
model_name = args.molmo_model
model_path = 'ghazishazan/VideoMolmo'
quant_config = BitsAndBytesConfig(
    load_in_4bit=True
)

model = MolmoForCausalLM.from_pretrained(
    model_path,
    torch_dtype='auto',
    device_map='auto',
    quantization_config=quant_config
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
processor = MultiModalPreprocessor(tokenizer=tokenizer)
model.eval()

predictor = build_sam2_video_predictor(args.sam2_config, args.sam2_checkpoint, device=device)

def get_output(images, prev_frames, prompt, model=None, processor=None, tokenizer=None):
    conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ""},
        ]
    prompt = tokenizer.apply_chat_template(
        conversation,
        chat_template=tokenizer.chat_template,
        add_generation_prompt=False,
        tokenize=False,
        return_dict=False
    )
    split_text = prompt.split("Assistant:")
    messages = [" " + split_text[0].strip() + " Assistant:", "" + split_text[1].strip()]
    
    images = pil_to_np(images)
    prev_frames = pil_to_np(prev_frames)

    inputs = processor(
        images=images,
        prev_frames=prev_frames,
        messages=messages,
    )
    # move inputs to the correct device and make a batch of size 1
    inputs = {k: torch.from_numpy(v).to(model.device).unsqueeze(0) for k, v in inputs.items()}
    inputs['input_ids'] = inputs['input_tokens'].clone()
    # generate output; maximum 200 new tokens; stop generation when 
        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
    with torch.autocast(device_type=str(device), enabled=True, dtype=torch.bfloat16):
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=300, stop_strings='<|endoftext|>'),
            tokenizer=processor.tokenizer
        )

    # only get generated tokens; decode them to text
    generated_tokens = output[0,inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"Output: {generated_text}")
    return generated_text


max_video_len = 100
points = {}
frames_dir = "examples/video_sample1"
video = os.path.basename(frames_dir)
print(f'video: {video}')

frames = sorted([frame for frame in os.listdir(frames_dir) if frame.endswith(('.jpg', '.png'))])
pil_images = [Image.open(os.path.join(frames_dir, frame)) for frame in frames]

total_frames = min(len(pil_images), max_video_len)
pil_images = pil_images[:total_frames]
print(f'total_frames: {total_frames}')

np_images = [np.array(image).astype(np.uint8) for image in pil_images]
save_images_folder(np_images, f'{video}')
with torch.inference_mode(), torch.autocast(str(device), dtype=torch.bfloat16):
    inference_state = predictor.init_state(video_path=f'{video}')
# shutil.rmtree(f'{video}')
prompt = "Point to the left hand"
print(f'prompt: {prompt}')
first_frame = pil_images[0]
w, h = first_frame.size
frame_rate = args.frame_rate
left_segments = {}
for f in range(0, total_frames, frame_rate):
    available = min(f, 4)
    prev_frames = (
        [Image.new(mode="RGB", size=(w, h), color=(0, 0, 0)) for _ in range(4 - available)] +
                [pil_images[f - i - 1] for i in range(available)]
    )
    image = pil_images[f]
    outputs = get_output([image], prev_frames, prompt, model=model, processor=processor, tokenizer=tokenizer)

    predictor.reset_state(inference_state)

    coords = get_coords(outputs, image=first_frame)
    points[f] = coords
    # if MOLMO fails to predict the point
    if coords == []:    
        # Populate video_segments with zeros
        for out_frame_idx in range(f, f + min(frame_rate, total_frames - f)):
            left_segments[out_frame_idx] = {
                out_obj_id: np.zeros((1, h, w), dtype=np.uint8)  # Use the appropriate shape (height, width)
                for out_obj_id in range(1)
            }
    
    else:  # MOLMO gets the points
        input_points = np.array(coords)
        input_labels = np.ones(len(input_points), dtype=np.int32)

        # Add all the points.
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

        max_frame_num_to_track = min(frame_rate, total_frames - f) - 1
        with torch.inference_mode(), torch.autocast(str(device), dtype=torch.bfloat16):
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                inference_state, max_frame_num_to_track=max_frame_num_to_track
            ):
                left_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
# save the predicted masks
save_path = os.path.join(args.save_path, f'{video}')
os.makedirs(save_path, exist_ok=True)
with open(f'{save_path}/points.jsonl', 'w') as f:
    f.write(json.dumps(points))

for frame in left_segments.keys():
    mask = np.zeros((h, w), dtype=np.uint8)
    for obj, curr_mask in left_segments[frame].items():
        mask = np.logical_or(mask, curr_mask[0])

    # Convert boolean mask to uint8
    mask = (mask * 255).astype(np.uint8)
    
    # Save with frame-specific filename
    frame_filename = os.path.join(save_path, f"{frames[frame].split('.')[0]}.png")
    cv2.imwrite(frame_filename, mask)