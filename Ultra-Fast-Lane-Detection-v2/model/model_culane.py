import torch
from model.backbone import resnet


# ============================================================
# Local weight initialization
# Avoids importing utils.common -> NVIDIA DALI
# ============================================================

def initialize_weights(*models):
    for model in models:
        for module in model.modules():

            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out"
                )

                if module.bias is not None:
                    torch.nn.init.constant_(
                        module.bias,
                        0
                    )

            elif isinstance(module, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(
                    module.weight,
                    1
                )

                torch.nn.init.constant_(
                    module.bias,
                    0
                )

            elif isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(
                    module.weight,
                    std=0.001
                )

                if module.bias is not None:
                    torch.nn.init.constant_(
                        module.bias,
                        0
                    )


# ============================================================
# UFLDv2 CULane Network
# ============================================================

class parsingNet(torch.nn.Module):

    def __init__(
        self,
        pretrained=True,
        backbone="18",
        num_grid_row=None,
        num_cls_row=None,
        num_grid_col=None,
        num_cls_col=None,
        num_lane_on_row=None,
        num_lane_on_col=None,
        use_aux=False,
        input_height=None,
        input_width=None,
        fc_norm=False
    ):

        super(parsingNet, self).__init__()

        self.num_grid_row = num_grid_row
        self.num_cls_row = num_cls_row

        self.num_grid_col = num_grid_col
        self.num_cls_col = num_cls_col

        self.num_lane_on_row = num_lane_on_row
        self.num_lane_on_col = num_lane_on_col

        self.use_aux = use_aux

        # ----------------------------------------------------
        # Output dimensions
        # ----------------------------------------------------

        self.dim1 = (
            self.num_grid_row
            * self.num_cls_row
            * self.num_lane_on_row
        )

        self.dim2 = (
            self.num_grid_col
            * self.num_cls_col
            * self.num_lane_on_col
        )

        self.dim3 = (
            2
            * self.num_cls_row
            * self.num_lane_on_row
        )

        self.dim4 = (
            2
            * self.num_cls_col
            * self.num_lane_on_col
        )

        self.total_dim = (
            self.dim1
            + self.dim2
            + self.dim3
            + self.dim4
        )

        mlp_mid_dim = 2048

        self.input_dim = (
            input_height // 32
            * input_width // 32
            * 8
        )

        # ----------------------------------------------------
        # Backbone
        # ----------------------------------------------------

        self.model = resnet(
            backbone,
            pretrained=pretrained
        )

        # ----------------------------------------------------
        # Classification Head
        # ----------------------------------------------------

        self.cls = torch.nn.Sequential(

            torch.nn.LayerNorm(self.input_dim)
            if fc_norm
            else torch.nn.Identity(),

            torch.nn.Linear(
                self.input_dim,
                mlp_mid_dim
            ),

            torch.nn.ReLU(),

            torch.nn.Linear(
                mlp_mid_dim,
                self.total_dim
            )
        )

        # ----------------------------------------------------
        # Feature pooling
        # ----------------------------------------------------

        if backbone in ["18", "34", "34fca"]:

            self.pool = torch.nn.Conv2d(
                512,
                8,
                1
            )

        else:

            self.pool = torch.nn.Conv2d(
                2048,
                8,
                1
            )

        # ----------------------------------------------------
        # We do NOT need auxiliary segmentation for inference.
        # Removing SegHead also removes the DALI dependency.
        # ----------------------------------------------------

        if self.use_aux:
            raise NotImplementedError(
                "Auxiliary segmentation is disabled "
                "for macOS inference."
            )

        initialize_weights(
            self.cls
        )


    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):

        x2, x3, fea = self.model(x)

        fea = self.pool(fea)

        fea = fea.view(
            -1,
            self.input_dim
        )

        out = self.cls(fea)

        pred_dict = {

            "loc_row":
                out[
                    :,
                    :self.dim1
                ].view(
                    -1,
                    self.num_grid_row,
                    self.num_cls_row,
                    self.num_lane_on_row
                ),

            "loc_col":
                out[
                    :,
                    self.dim1:
                    self.dim1 + self.dim2
                ].view(
                    -1,
                    self.num_grid_col,
                    self.num_cls_col,
                    self.num_lane_on_col
                ),

            "exist_row":
                out[
                    :,
                    self.dim1 + self.dim2:
                    self.dim1 + self.dim2 + self.dim3
                ].view(
                    -1,
                    2,
                    self.num_cls_row,
                    self.num_lane_on_row
                ),

            "exist_col":
                out[
                    :,
                    -self.dim4:
                ].view(
                    -1,
                    2,
                    self.num_cls_col,
                    self.num_lane_on_col
                )
        }

        return pred_dict


    # ========================================================
    # Test-Time Augmentation
    # ========================================================

    def forward_tta(self, x):

        x2, x3, fea = self.model(x)

        pooled_fea = self.pool(fea)

        n, c, h, w = pooled_fea.shape

        left_pooled_fea = torch.zeros_like(
            pooled_fea
        )

        right_pooled_fea = torch.zeros_like(
            pooled_fea
        )

        up_pooled_fea = torch.zeros_like(
            pooled_fea
        )

        down_pooled_fea = torch.zeros_like(
            pooled_fea
        )

        left_pooled_fea[:, :, :, :w - 1] = (
            pooled_fea[:, :, :, 1:]
        )

        left_pooled_fea[:, :, :, -1] = (
            pooled_fea.mean(-1)
        )

        right_pooled_fea[:, :, :, 1:] = (
            pooled_fea[:, :, :, :w - 1]
        )

        right_pooled_fea[:, :, :, 0] = (
            pooled_fea.mean(-1)
        )

        up_pooled_fea[:, :, :h - 1, :] = (
            pooled_fea[:, :, 1:, :]
        )

        up_pooled_fea[:, :, -1, :] = (
            pooled_fea.mean(-2)
        )

        down_pooled_fea[:, :, 1:, :] = (
            pooled_fea[:, :, :h - 1, :]
        )

        down_pooled_fea[:, :, 0, :] = (
            pooled_fea.mean(-2)
        )

        fea = torch.cat(
            [
                pooled_fea,
                left_pooled_fea,
                right_pooled_fea,
                up_pooled_fea,
                down_pooled_fea
            ],
            dim=0
        )

        fea = fea.view(
            -1,
            self.input_dim
        )

        out = self.cls(fea)

        return {

            "loc_row":
                out[
                    :,
                    :self.dim1
                ].view(
                    -1,
                    self.num_grid_row,
                    self.num_cls_row,
                    self.num_lane_on_row
                ),

            "loc_col":
                out[
                    :,
                    self.dim1:
                    self.dim1 + self.dim2
                ].view(
                    -1,
                    self.num_grid_col,
                    self.num_cls_col,
                    self.num_lane_on_col
                ),

            "exist_row":
                out[
                    :,
                    self.dim1 + self.dim2:
                    self.dim1 + self.dim2 + self.dim3
                ].view(
                    -1,
                    2,
                    self.num_cls_row,
                    self.num_lane_on_row
                ),

            "exist_col":
                out[
                    :,
                    -self.dim4:
                ].view(
                    -1,
                    2,
                    self.num_cls_col,
                    self.num_lane_on_col
                )
        }


# ============================================================
# Model Factory
# ============================================================

def get_model(cfg):

    model = parsingNet(

        pretrained=True,

        backbone=cfg.backbone,

        num_grid_row=cfg.num_cell_row,

        num_cls_row=cfg.num_row,

        num_grid_col=cfg.num_cell_col,

        num_cls_col=cfg.num_col,

        num_lane_on_row=cfg.num_lanes,

        num_lane_on_col=cfg.num_lanes,

        use_aux=False,

        input_height=cfg.train_height,

        input_width=cfg.train_width,

        fc_norm=cfg.fc_norm
    )

    # Apple Silicon
    if torch.backends.mps.is_available():

        model = model.to("mps")

    # NVIDIA / CUDA
    elif torch.cuda.is_available():

        model = model.cuda()

    # Otherwise CPU
    else:

        model = model.cpu()

    return model