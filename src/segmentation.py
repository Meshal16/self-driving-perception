import torch
import cv2
import numpy as np
import torch.nn.functional as F

from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation
)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")

elif torch.cuda.is_available():
    device = torch.device("cuda")

else:
    device = torch.device("cpu")


print(f"SegFormer device: {device}")


# ============================================================
# MODEL
# ============================================================

model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

processor = SegformerImageProcessor.from_pretrained(
    model_name
)

model = SegformerForSemanticSegmentation.from_pretrained(
    model_name
)


# Move model to GPU / MPS / CPU
model = model.to(device)

model.eval()


print(f"SegFormer ready on {device}")


# ============================================================
# ROAD MASK CLEANUP
# ============================================================

def clean_road_mask(road_mask):

    height, width = road_mask.shape


    # --------------------------------------------------------
    # 1. MORPHOLOGICAL CLEANUP
    # --------------------------------------------------------
    # Removes tiny noise and fills small gaps

    kernel = np.ones(
        (7, 7),
        np.uint8
    )


    cleaned = cv2.morphologyEx(
        road_mask,
        cv2.MORPH_OPEN,
        kernel
    )


    cleaned = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        kernel
    )


    # --------------------------------------------------------
    # 2. REMOVE SMALL ROAD BLOBS
    # --------------------------------------------------------

    number_of_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            cleaned,
            connectivity=8
        )
    )


    filtered_mask = np.zeros_like(
        cleaned
    )


    minimum_area = int(
        width * height * 0.002
    )


    for label_index in range(
        1,
        number_of_labels
    ):

        area = stats[
            label_index,
            cv2.CC_STAT_AREA
        ]


        if area >= minimum_area:

            filtered_mask[
                labels == label_index
            ] = 255


    # --------------------------------------------------------
    # 3. REMOVE VERY BOTTOM OF FRAME
    # --------------------------------------------------------
    # Last 12% is ignored because this area often contains
    # the dashcam vehicle hood.

    hood_start = int(
        height * 0.88
    )


    filtered_mask[
        hood_start:height,
        :
    ] = 0


    return filtered_mask


# ============================================================
# SEGMENT ROAD
# ============================================================

def segment_road(image_input):

    # --------------------------------------------------------
    # 1. IMAGE PATH OR VIDEO FRAME
    # --------------------------------------------------------

    if isinstance(image_input, str):

        image_bgr = cv2.imread(
            image_input
        )


        if image_bgr is None:

            raise ValueError(
                f"Could not read image: {image_input}"
            )

    else:

        image_bgr = image_input.copy()


    # --------------------------------------------------------
    # 2. BGR -> RGB
    # --------------------------------------------------------

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )


    height, width = image_rgb.shape[:2]


    # --------------------------------------------------------
    # 3. SEGFORMER PREPROCESSING
    # --------------------------------------------------------

    inputs = processor(
        images=image_rgb,
        return_tensors="pt"
    )


    # --------------------------------------------------------
    # 4. MOVE INPUTS TO SAME DEVICE AS MODEL
    # --------------------------------------------------------

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }


    # --------------------------------------------------------
    # 5. SEGFORMER INFERENCE
    # --------------------------------------------------------

    with torch.inference_mode():

        outputs = model(
            **inputs
        )


    # --------------------------------------------------------
    # 6. RESIZE OUTPUT TO ORIGINAL FRAME SIZE
    # --------------------------------------------------------

    upsampled = F.interpolate(
        outputs.logits,
        size=(height, width),
        mode="bilinear",
        align_corners=False
    )


    # --------------------------------------------------------
    # 7. GET CLASS FOR EACH PIXEL
    # --------------------------------------------------------

    predicted_class = upsampled.argmax(
        dim=1
    )


    # ========================================================
    # ROAD
    # Cityscapes class 0
    # ========================================================

    road_mask = (
        predicted_class[0] == 0
    )


    road_mask_img = (
        road_mask
        .cpu()
        .numpy()
        .astype("uint8")
        * 255
    )


    # ========================================================
    # CLEAN ROAD MASK
    # ========================================================

    road_mask_img = clean_road_mask(
        road_mask_img
    )


    # ========================================================
    # SIDEWALK
    # Cityscapes class 1
    # ========================================================

    sidewalk_mask = (
        predicted_class[0] == 1
    )


    sidewalk_mask_img = (
        sidewalk_mask
        .cpu()
        .numpy()
        .astype("uint8")
        * 255
    )


    # ========================================================
    # VISUALIZATION
    # ========================================================

    display_image = image_bgr.copy()


    colored_mask = np.zeros_like(
        display_image
    )


    # Road = Green
    colored_mask[
        road_mask_img == 255
    ] = (
        0,
        255,
        0
    )


    # Sidewalk = Red
    colored_mask[
        sidewalk_mask_img == 255
    ] = (
        0,
        0,
        255
    )


    # ========================================================
    # OVERLAY
    # ========================================================

    overlay = cv2.addWeighted(
        display_image,
        1.0,
        colored_mask,
        0.2,
        0
    )


    return overlay