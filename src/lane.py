import os
import sys
import cv2
import torch
import numpy as np

from PIL import Image
from torchvision import transforms


# ============================================================
# CONNECT UFLDv2 TO MAIN PROJECT
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

UFLD_PATH = os.path.join(
    PROJECT_ROOT,
    "Ultra-Fast-Lane-Detection-v2"
)

sys.path.insert(0, UFLD_PATH)

from model.model_culane import parsingNet


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ============================================================
# CULANE CONFIG
# ============================================================

NUM_GRID_ROW = 200
NUM_CLS_ROW = 72

NUM_GRID_COL = 100
NUM_CLS_COL = 81

NUM_LANES = 4

INPUT_HEIGHT = 320
INPUT_WIDTH = 1600

CROP_RATIO = 0.6


ROW_ANCHOR = np.linspace(
    0.42,
    1.0,
    NUM_CLS_ROW
)

COL_ANCHOR = np.linspace(
    0.0,
    1.0,
    NUM_CLS_COL
)


# ============================================================
# BUILD MODEL
# ============================================================

print("Loading UFLDv2 lane model...")


lane_model = parsingNet(
    pretrained=False,
    backbone="18",

    num_grid_row=NUM_GRID_ROW,
    num_cls_row=NUM_CLS_ROW,

    num_grid_col=NUM_GRID_COL,
    num_cls_col=NUM_CLS_COL,

    num_lane_on_row=NUM_LANES,
    num_lane_on_col=NUM_LANES,

    use_aux=False,

    input_height=INPUT_HEIGHT,
    input_width=INPUT_WIDTH,

    fc_norm=True
)


# ============================================================
# LOAD PRETRAINED WEIGHTS
# ============================================================

weights_path = os.path.join(
    UFLD_PATH,
    "weights",
    "culane_res18.pth"
)


checkpoint = torch.load(
    weights_path,
    map_location="cpu"
)


if isinstance(checkpoint, dict) and "model" in checkpoint:
    state_dict = checkpoint["model"]
else:
    state_dict = checkpoint


clean_state_dict = {}

for key, value in state_dict.items():

    if key.startswith("module."):
        key = key[7:]

    clean_state_dict[key] = value


lane_model.load_state_dict(
    clean_state_dict,
    strict=True
)


lane_model = lane_model.to(device)

lane_model.eval()


print(
    f"UFLDv2 lane model ready on {device}"
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

resize_height = int(
    INPUT_HEIGHT / CROP_RATIO
)


lane_transform = transforms.Compose([
    transforms.Resize(
        (resize_height, INPUT_WIDTH)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])


# ============================================================
# DECODE MODEL OUTPUT
# ============================================================

def pred2coords(
    pred,
    image_width,
    image_height,
    local_width=1
):

    loc_row = pred["loc_row"].detach().cpu()
    loc_col = pred["loc_col"].detach().cpu()

    exist_row = pred["exist_row"].detach().cpu()
    exist_col = pred["exist_col"].detach().cpu()


    max_indices_row = loc_row.argmax(1)
    max_indices_col = loc_col.argmax(1)

    valid_row = exist_row.argmax(1)
    valid_col = exist_col.argmax(1)


    lanes = []


    # ========================================================
    # ROW-BASED LANES
    # ========================================================

    for lane_index in [1, 2]:

        lane_points = []


        if (
            valid_row[0, :, lane_index].sum()
            > NUM_CLS_ROW / 2
        ):

            for k in range(NUM_CLS_ROW):

                if valid_row[0, k, lane_index]:

                    center = int(
                        max_indices_row[
                            0,
                            k,
                            lane_index
                        ]
                    )


                    start = max(
                        0,
                        center - local_width
                    )

                    end = min(
                        NUM_GRID_ROW - 1,
                        center + local_width
                    )


                    indices = torch.arange(
                        start,
                        end + 1
                    )


                    probabilities = torch.softmax(
                        loc_row[
                            0,
                            indices,
                            k,
                            lane_index
                        ],
                        dim=0
                    )


                    position = (
                        probabilities
                        * indices.float()
                    ).sum() + 0.5


                    x = (
                        position
                        / (NUM_GRID_ROW - 1)
                        * image_width
                    )


                    y = (
                        ROW_ANCHOR[k]
                        * image_height
                    )


                    lane_points.append(
                        (
                            int(x),
                            int(y)
                        )
                    )


        lanes.append(lane_points)


    # ========================================================
    # COLUMN-BASED LANES
    # ========================================================

    for lane_index in [0, 3]:

        lane_points = []


        if (
            valid_col[0, :, lane_index].sum()
            > NUM_CLS_COL / 4
        ):

            for k in range(NUM_CLS_COL):

                if valid_col[0, k, lane_index]:

                    center = int(
                        max_indices_col[
                            0,
                            k,
                            lane_index
                        ]
                    )


                    start = max(
                        0,
                        center - local_width
                    )

                    end = min(
                        NUM_GRID_COL - 1,
                        center + local_width
                    )


                    indices = torch.arange(
                        start,
                        end + 1
                    )


                    probabilities = torch.softmax(
                        loc_col[
                            0,
                            indices,
                            k,
                            lane_index
                        ],
                        dim=0
                    )


                    position = (
                        probabilities
                        * indices.float()
                    ).sum() + 0.5


                    y = (
                        position
                        / (NUM_GRID_COL - 1)
                        * image_height
                    )


                    x = (
                        COL_ANCHOR[k]
                        * image_width
                    )


                    lane_points.append(
                        (
                            int(x),
                            int(y)
                        )
                    )


        lanes.append(lane_points)


    return lanes


# ============================================================
# MAIN LANE DETECTION FUNCTION
# ============================================================

def lane_detection(
    image_input,
    base_image=None
):

    # ========================================================
    # READ IMAGE OR RECEIVE VIDEO FRAME
    # ========================================================

    if isinstance(image_input, str):

        # Image path
        original_image = cv2.imread(
            image_input
        )

        if original_image is None:
            raise FileNotFoundError(
                f"Could not read image: {image_input}"
            )

    else:

        # Video frame
        original_image = image_input.copy()


    height, width = original_image.shape[:2]


    # ========================================================
    # PREPROCESS FOR UFLDv2
    # ========================================================

    rgb_image = cv2.cvtColor(
        original_image,
        cv2.COLOR_BGR2RGB
    )


    pil_image = Image.fromarray(
        rgb_image
    )


    image_tensor = lane_transform(
        pil_image
    )


    # Keep bottom section expected by CULane model
    image_tensor = image_tensor[
        :,
        -INPUT_HEIGHT:,
        :
    ]


    # CHW -> BCHW
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(
        device
    )


    # ========================================================
    # AI INFERENCE
    # ========================================================

    with torch.no_grad():

        prediction = lane_model(
            image_tensor
        )


    # ========================================================
    # DECODE PREDICTIONS
    # ========================================================

    lanes = pred2coords(
        prediction,
        width,
        height
    )


    # ========================================================
    # BASE IMAGE
    # ========================================================

    if base_image is None:

        result = original_image.copy()

    else:

        result = base_image.copy()


    # ========================================================
    # DRAW AI-PREDICTED LANES
    # ========================================================

    for lane in lanes:

        if len(lane) < 2:
            continue


        points = np.array(
            lane,
            dtype=np.int32
        )


        # Smooth connected visualization
        cv2.polylines(
            result,
            [points],
            False,
            (255, 0, 0),
            4
        )


        # Predicted anchor points
        for point in lane:

            cv2.circle(
                result,
                point,
                3,
                (0, 255, 255),
                -1
            )


    return result, lanes