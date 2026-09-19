import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import cv2
import torch.nn.functional as F


model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

processor = SegformerImageProcessor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name)


def segment_road(image_path):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    upsampled = F.interpolate(
        outputs.logits,
        size=(height, width),
        mode="bilinear",
        align_corners=False
    )

    predicted_class = upsampled.argmax(dim=1)

    # Road
    road_mask = (predicted_class[0] == 0)
    road_mask_img = road_mask.cpu().numpy().astype("uint8") * 255

    # Sidewalk
    sidewalk_mask = (predicted_class[0] == 1)
    sidewalk_mask_img = sidewalk_mask.cpu().numpy().astype("uint8") * 255

    # Prepare image for visualization
    display_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Create colored mask
    colored_mask = display_image.copy()
    colored_mask[:] = 0

    colored_mask[road_mask_img == 255] = (0, 255, 0)
    colored_mask[sidewalk_mask_img == 255] = (0, 0, 255)

    # Overlay
    overlay = cv2.addWeighted(
        display_image,
        1.0,
        colored_mask,
        0.2,
        0
    )

    return overlay