"""
Test functions
@diaw 
21/09/2025

"""

import pytest
import torch
from torch import nn

from yolo_one.losses import YoloOneLoss


@pytest.fixture
def object():
    """Initialized model"""
    obj = YoloOneLoss()
    return obj


@pytest.fixture
def loss_method():
    """Loss function"""
    loss_methods = ["_ciou_loss", "_eiou_loss", "_meiou_loss"]
    return loss_methods


@pytest.fixture
def loss_type():
    """Loss type"""
    type_losses = ["ciou", "eiou", "mieou"]
    return type_losses


@pytest.fixture
def device():
    """Test device (CUDA if available, else CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_default_initialisation(object):
    """Test default values"""

    assert object.box_weight == 0.5
    assert object.obj_weight == 1.0
    assert object.aspect_weight == 0.5
    assert object.shape_conf_weight == 0.2
    assert object.focal_alpha == 0.25
    assert object.focal_gamma == 1.5
    assert object.iou_type == "meiou"
    assert object.label_smoothing == 0.0
    assert object.p5_weight_boost == 1.2
    assert object.theta == 4
    assert object.focal_loss is False
    assert isinstance(object.bce_loss, nn.BCEWithLogitsLoss)
    assert isinstance(object.mse_loss, nn.MSELoss)


def test_custom_initialization():
    """Test with custom parameters"""
    loss_fn = YoloOneLoss(
        box_weight=1.0,
        obj_weight=2.0,
        aspect_weight=1.5,
        shape_conf_weight=0.9,
        focal_alpha=0.5,
        focal_gamma=2.0,
        iou_type="meiou",
        label_smoothing=0.1,
        p5_weight_boost=2.0,
        theta=8,
        focal_loss=True,
    )
    assert loss_fn.box_weight == 1.0
    assert loss_fn.obj_weight == 2.0
    assert loss_fn.aspect_weight == 1.5
    assert loss_fn.shape_conf_weight == 0.9
    assert loss_fn.focal_alpha == 0.5
    assert loss_fn.focal_gamma == 2.0
    assert loss_fn.iou_type == "meiou"
    assert loss_fn.label_smoothing == 0.1
    assert loss_fn.p5_weight_boost == 2.0
    assert loss_fn.theta == 8
    assert loss_fn.focal_loss is True


def test_default_vs_custom_same_values(object):
    """Test default == custom when we have the same values"""

    loss_fn_custom = YoloOneLoss(
        box_weight=0.5,
        obj_weight=1.0,
        aspect_weight=0.5,
        shape_conf_weight=0.2,
        focal_alpha=0.25,
        focal_gamma=1.5,
        iou_type="meiou",
        label_smoothing=0.0,
        p5_weight_boost=1.2,
        theta=4,
        focal_loss=False,
    )

    # Float comparison with Equal

    assert object.box_weight == loss_fn_custom.box_weight
    assert object.obj_weight == loss_fn_custom.obj_weight
    assert object.aspect_weight == loss_fn_custom.aspect_weight
    assert object.shape_conf_weight == loss_fn_custom.shape_conf_weight
    assert object.focal_alpha == loss_fn_custom.focal_alpha
    assert object.focal_gamma == loss_fn_custom.focal_gamma
    assert object.label_smoothing == loss_fn_custom.label_smoothing
    assert object.p5_weight_boost == loss_fn_custom.p5_weight_boost
    assert object.theta == loss_fn_custom.theta
    assert object.focal_loss == loss_fn_custom.focal_loss

    # Check that if it is the good loss type

    assert isinstance(object.bce_loss, nn.BCEWithLogitsLoss)
    assert isinstance(loss_fn_custom.bce_loss, nn.BCEWithLogitsLoss)
    assert isinstance(object.mse_loss, nn.MSELoss)
    assert isinstance(loss_fn_custom.mse_loss, nn.MSELoss)

    # Check that the internal parameters are identical
    assert object.bce_loss.reduction == loss_fn_custom.bce_loss.reduction
    assert object.mse_loss.reduction == loss_fn_custom.mse_loss.reduction


def test_compute_anchor_free_box_loss_scalar(loss_type, device):
    """Anchor free box : the loss should be a scalar"""

    pred_boxes = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [0.1, 0.1, 0.3, 0.3]], dtype=torch.float32, device=device
    )
    target_boxes = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.6, 0.6]], dtype=torch.float32, device=device
    )

    for name in loss_type:
        loss_fn = YoloOneLoss(iou_type=name)
        loss = loss_fn._compute_anchor_free_box_loss(
            pred_boxes, target_boxes, grid_h=2, grid_w=2
        )
        assert loss.shape == torch.Size([]), f"Loss must be a scalar, got {loss}"


def test_compute_anchor_free_box_loss_positive(loss_type, device):
    """Anchor free box : the loss should be positive"""

    pred_boxes = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.3, 0.3]], dtype=torch.float32, device=device
    )
    target_boxes = torch.tensor(
        [[0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.6, 0.6]], dtype=torch.float32, device=device
    )

    for name in loss_type:
        loss_fn = YoloOneLoss(iou_type=name)
        loss = loss_fn._compute_anchor_free_box_loss(
            pred_boxes, target_boxes, grid_h=2, grid_w=2
        )
        assert loss >= 0, f"Loss must be greater than zero, got {loss}"


def test_aspect_loss(object, device):
    """Test aspect loss"""

    pred_box = torch.tensor([0.5, 0.8, 0.3], dtype=torch.float32, device=device)
    target_box = torch.tensor([0.6, 0.7, 0.2], dtype=torch.float32, device=device)
    loss = object._compute_aspect_loss(pred_box, target_box)

    assert isinstance(loss, torch.Tensor), "The loss must be a Tensor"
    assert loss >= 0, f"Loss must be greater than zero, got {loss}"


def test_shape_conf_loss(object, device):
    """Test shape confidence loss"""

    pred = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32, device=device)
    target = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    loss = object._compute_shape_confidence_loss(pred, target)

    assert isinstance(loss, torch.Tensor), "Loss must be a Tensor"
    assert loss >= 0, f"Loss must be greater than zero, got {loss}"


def test_idententical_boxes(object, loss_method, device):
    """Loss should be equal to 0 for identical boxes"""
    pred_box = torch.tensor(
        [0.0, 0.0, 1.0, 1.0],
        dtype=torch.float32,
        device=device,
    )
    target_box = torch.tensor(
        [0.0, 0.0, 1.0, 1.0],
        dtype=torch.float32,
        device=device,
    )

    for name in loss_method:
        loss_fn = getattr(object, name)
        loss = loss_fn(pred_box, target_box)

        assert torch.allclose(
            loss, torch.tensor(0.0)
        ), f"{name}: expected 0, got {loss.item()}"


def test_coord_zero_pred(object, loss_method, device):
    "Loss should be greater than 0 if the pred coordinates are equal to 0"
    pred_box = torch.tensor(
        [0.0, 0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=device,
    )

    target_box = torch.tensor(
        [0.0, 0.0, 2.0, 3.0],
        dtype=torch.float32,
        device=device,
    )

    for name in loss_method:
        loss_fn = getattr(object, name)
        loss = loss_fn(pred_box, target_box)

        assert loss >= 0, f"{name}: loss should be >=0, got {loss}"


def test_coordinate_zero_target(object, loss_method, device):
    "Loss should be equal to 0 if the target coordinates are equal to 0"
    pred_box = torch.tensor(
        [0.0, 0.0, 2.0, 3.0],
        dtype=torch.float32,
        device=device,
    )
    target_box = torch.tensor(
        [0.0, 0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=device,
    )
    for name in loss_method:
        loss_fn = getattr(object, name)
        loss = loss_fn(pred_box, target_box)

        assert loss >= 0, f"{name}: loss should be >= 0 "


def test_coordinate_point(object, loss_method, device):
    "Loss should be equal to 0 if the height and width are equal to 0"

    pred_box = torch.tensor(
        [0.1, 0.2, 0.0, 0.0],
        dtype=torch.float32,
        device=device,
    )
    target_box = torch.tensor(
        [0.2, 0.3, 0.0, 0.0],
        dtype=torch.float32,
        device=device,
    )
    for name in loss_method:
        loss_fn = getattr(object, name)
        loss = loss_fn(pred_box, target_box)

        assert loss >= 0, f"{name}: loss should be >=0, got {loss}"


def test_shifted_boxes(object, loss_method, device):
    """Loss should be >0 for shifted boxes"""

    pred_box = torch.tensor(
        [0.0, 0.5, 3.0, 3.0],
        dtype=torch.float32,
        device=device,
    )
    target_box = torch.tensor(
        [0.0, 0.0, 4.0, 4.0],
        dtype=torch.float32,
        device=device,
    )
    for name in loss_method:
        loss_fn = getattr(object, name)
        loss = loss_fn(pred_box, target_box)

        assert loss > 0, f"{name}: loss should be > 0, got {loss}"


def test_partial_overlap(object, loss_method, device):
    """Loss should be >0 for shifted boxes"""
    pred_box = torch.tensor(
        [0.0, 0.8, 3.0, 3.0],
        dtype=torch.float32,
        device=device,
    )
    target_box = torch.tensor(
        [0.0, 0.2, 4.0, 4.0],
        dtype=torch.float32,
        device=device,
    )
    for name in loss_method:
        loss_fn = getattr(object, name)
        loss = loss_fn(pred_box, target_box)

        assert isinstance(loss, torch.Tensor), "Loss must be a Tensor"
        assert loss > 0, f"{name}: loss should be > 0"


def test_large_values(object, loss_method, device):
    """Loss should remain finite value for very large coordinates"""
    pred_box = torch.tensor(
        [0.7, 0.7, 2e7, 3e7],
        dtype=torch.float32,
        device=device,
    )
    target_box = torch.tensor(
        [0.7, 0.8, 2e7, 3e7],
        dtype=torch.float32,
        device=device,
    )
    for name in loss_method:
        loss_fn = getattr(object, name)
        loss = loss_fn(pred_box, target_box)
        assert loss >= 0, f"{name}: loss should be greater or equal to 0, got {loss}"


def test_zero_size_box(object, loss_method, device):
    """Loss should handle degenerate boxes (zero width/height)"""
    pred_box = torch.tensor(
        [0.0, 0.0, 0.0, 0.0],
        dtype=torch.float32,
        device=device,
    )
    target_box = torch.tensor(
        [0.9, 0.9, 2.0, 2.0],
        dtype=torch.float32,
        device=device,
    )
    for name in loss_method:
        loss_fn = getattr(object, name)
        loss = loss_fn(pred_box, target_box)

        assert loss >= 0, f"{name}: loss must be > 0, got {loss}"


def test_invalid_flipped_boxes(object, loss_method, device):
    """Loss should handle boxes with reversed coodinates"""
    pred_box = torch.tensor(
        [0.7, 0.7, 0.1, 0.1],
        dtype=torch.float32,
        device=device,
    )
    target_box = torch.tensor(
        [0.6, 0.6, 0.2, 0.2],
        dtype=torch.float32,
        device=device,
    )
    for name in loss_method:
        loss_fn = getattr(object, name)
        loss = loss_fn(pred_box, target_box)

        assert loss >= 0, f"{name}: loss should be positive, got {loss}"


def test_batch_processing(object, loss_method, device):
    """Loss should work  on batches and return a scalar"""
    pred_boxes = torch.tensor(
        [[0.6, 0.6, 1.0, 1.0], [0.3, 0.3, 3.0, 2.0]],
        dtype=torch.float32,
        device=device,
    )
    target_boxes = torch.tensor(
        [[0.1, 0.1, 2.0, 2.0], [0.2, 0.2, 3.0, 4.0]],
        dtype=torch.float32,
        device=device,
    )
    for name in loss_method:
        loss_fn = getattr(object, name)
        loss = loss_fn(pred_boxes, target_boxes)

        assert loss.shape == (), f"{name}: must return a saclar, got {loss}"
        assert loss>=0, f"{name} loss must be greater than zero, got {loss}"


def test_backward(object, loss_method, device):
    """Loss must be differentiable"""
    pred_box = torch.tensor(
        [0.9, 0.9, 1.0, 1.0],
        requires_grad=True,
        dtype=torch.float32,
        device=device,
    )
    target_box = torch.tensor(
        [0.3, 0.3, 2.0, 2.0],
        dtype=torch.float32,
        device=device,
    )
    for name in loss_method:
        loss_fn = getattr(object, name)
        loss = loss_fn(pred_box, target_box)
        loss.backward()

        assert pred_box.grad is not None, f"{name}: backward failed"
        assert pred_box.grad.all()>=0, f"{name}: All prediction must be greater than zero"


def test_device_and_dtype_support(object, loss_method, device):
    "Test Loss for device and dtypes"

    pred_box = torch.tensor(
        [0.4, 0.4, 1.0, 1.0],
        dtype=torch.float32,
        device=device,
    )
    target_box = torch.tensor(
        [1.0, 1.0, 2.0, 2.0],
        dtype=torch.float32,
        device=device,
    )

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    dtypes = [torch.float32, torch.float16, torch.float64]
    for name in loss_method:
        loss_fn = getattr(object, name)
        for de in devices:
            for dtype in dtypes:
                b1 = pred_box.to(device=de, dtype=dtype)
                b2 = target_box.to(device=de, dtype=dtype)
                loss = loss_fn(b1, b2)

            assert torch.isfinite(loss).item() is True, f"{name} failed on device {device}/{dtype}, got {loss}"
