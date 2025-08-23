import unittest
import torch
import torch.nn as nn


from yolo_one.losses import YoloOneLoss


class TestBoxLoss(unittest.TestCase):

    

    def setUp(self):

        """List of losses to test"""
        self.obj = YoloOneLoss()
        self.loss_methods = ["_ciou_loss", "_eiou_loss", "_meiou_loss", "_siou_loss"]
        self.chosen_loss = "meiou"
        self.type_loss = ["ciou", "eiou", "mieou", "siou"]
    def test_default_initialisation(self):

        """Test default values"""

        loss_fn = YoloOneLoss()
        self.assertEqual(loss_fn.box_weight, 7.5)
        self.assertEqual(loss_fn.obj_weight, 1.0)
        self.assertEqual(loss_fn.aspect_weight, 0.5)
        self.assertEqual(loss_fn.shape_conf_weight, 0.2)
        self.assertEqual(loss_fn.focal_alpha, 0.25)
        self.assertEqual(loss_fn.focal_gamma, 1.5)
        self.assertEqual(loss_fn.iou_type, self.chosen_loss)
        self.assertEqual(loss_fn.label_smoothing, 0.0)
        self.assertEqual(loss_fn.p5_weight_boost, 1.2)
        self.assertEqual(loss_fn.theta, 4)
        self.assertFalse(loss_fn.focal_loss)
        self.assertIsInstance(loss_fn.bce_loss, nn.BCEWithLogitsLoss)
        self.assertIsInstance(loss_fn.mse_loss, nn.MSELoss)

    
    def test_custom_initialization(self):
        """Test with custom parameters"""
        loss_fn = YoloOneLoss(
            box_weight=10.0,
            obj_weight=2.0,
            aspect_weight=1.5,
            shape_conf_weight=0.9,
            focal_alpha=0.5,
            focal_gamma=2.0,
            iou_type=self.chosen_loss,
            label_smoothing=0.1,
            p5_weight_boost=2.0,
            theta=8,
            focal_loss=True
        )
        self.assertEqual(loss_fn.box_weight, 10.0)
        self.assertEqual(loss_fn.obj_weight, 2.0)
        self.assertEqual(loss_fn.aspect_weight, 1.5)
        self.assertEqual(loss_fn.shape_conf_weight, 0.9)
        self.assertEqual(loss_fn.focal_alpha, 0.5)
        self.assertEqual(loss_fn.focal_gamma, 2.0)
        self.assertEqual(loss_fn.iou_type, self.chosen_loss)
        self.assertEqual(loss_fn.label_smoothing, 0.1)
        self.assertEqual(loss_fn.p5_weight_boost, 2.0)
        self.assertEqual(loss_fn.theta, 8)
        self.assertTrue(loss_fn.focal_loss) 

    def test_default_vs_custom_same_values(self):
        """Test default == custom when we have the same values"""
        loss_fn_default = YoloOneLoss()
        loss_fn_custom = YoloOneLoss(
            box_weight=7.5,
            obj_weight=1.0,
            aspect_weight=0.5,
            shape_conf_weight=0.2,
            focal_alpha=0.25,
            focal_gamma=1.5,
            iou_type=self.chosen_loss,
            label_smoothing=0.0,
            p5_weight_boost=1.2,
            theta=4,
            focal_loss=False
        )

        # Float comparison with Equal
        self.assertEqual(loss_fn_default.box_weight, loss_fn_custom.box_weight)
        self.assertEqual(loss_fn_default.obj_weight, loss_fn_custom.obj_weight)
        self.assertEqual(loss_fn_default.aspect_weight, loss_fn_custom.aspect_weight)
        self.assertEqual(loss_fn_default.shape_conf_weight, loss_fn_custom.shape_conf_weight)
        self.assertEqual(loss_fn_default.focal_alpha, loss_fn_custom.focal_alpha)
        self.assertEqual(loss_fn_default.focal_gamma, loss_fn_custom.focal_gamma)
        self.assertEqual(loss_fn_default.iou_type, loss_fn_custom.iou_type)
        self.assertEqual(loss_fn_default.label_smoothing, loss_fn_custom.label_smoothing)
        self.assertEqual(loss_fn_default.p5_weight_boost, loss_fn_custom.p5_weight_boost)
        self.assertEqual(loss_fn_default.theta, loss_fn_custom.theta)
        self.assertEqual(loss_fn_default.focal_loss, loss_fn_custom.focal_loss)

        # Check that if it is the good loss type 
        self.assertIsInstance(loss_fn_default.bce_loss, nn.BCEWithLogitsLoss)
        self.assertIsInstance(loss_fn_custom.bce_loss, nn.BCEWithLogitsLoss)
        self.assertIsInstance(loss_fn_default.mse_loss, nn.MSELoss)
        self.assertIsInstance(loss_fn_custom.mse_loss, nn.MSELoss)

        # Check that the internal parameters are identical
        self.assertEqual(loss_fn_default.bce_loss.reduction, loss_fn_custom.bce_loss.reduction)
        self.assertEqual(loss_fn_default.mse_loss.reduction, loss_fn_custom.mse_loss.reduction)


    def test_compute_anchor_free_box_loss_scalar(self):

        pred_boxes = torch.tensor([[0.0, 0.0, 0.0, 0.0],
                                        [1.0, 1.0, 0.3, 0.3]], dtype=torch.float32)
        target_boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0],
                                          [1.0, 1.0, 0.6, 0.6]], dtype=torch.float32)
        

        for name in self.type_loss:
            loss_fn = YoloOneLoss(iou_type=name)
            loss = loss_fn._compute_anchor_free_box_loss(pred_boxes, target_boxes, stride=1, grid_h=2, grid_w=2)
            self.assertEqual(loss.shape, torch.Size([]))

    def test_compute_anchor_free_box_loss_positive(self):

        pred_boxes = torch.tensor([[0.0, 0.0, 0.0, 0.0],
                                        [1.0, 1.0, 0.3, 0.3]], dtype=torch.float32)
        target_boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0],
                                          [1.0, 1.0, 0.6, 0.6]], dtype=torch.float32)

        for name in self.type_loss:
            loss_fn = YoloOneLoss(iou_type=name)
            loss = loss_fn._compute_anchor_free_box_loss(pred_boxes, target_boxes, stride=1, grid_h=2, grid_w=2)
            self.assertGreaterEqual(loss.item(), 0)

    def test_aspect_loss(self):
 
        pred_box = torch.tensor([0.5, 0.8, 0.3])
        target_box = torch.tensor([0.6, 0.7, 0.2])
        loss = self.obj._compute_aspect_loss(pred_box, target_box)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # must be scalar 
        self.assertGreaterEqual(loss.item(), 0.0)


    def test_shape_conf_loss(self):

        pred = torch.tensor([0.0, 1.0, -1.0])
        target = torch.tensor([0.0, 1.0, 0.0])
        loss = self.obj._compute_shape_confidence_loss(pred, target)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertGreaterEqual(loss.item(), 0.0)

    

    
    def test_idententical_boxes(self):
        """Loss should be equal to 0 for identical boxes"""
        pred_box = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        target_box = torch.tensor([[0.0, 0.0, 1.0, 1.0]])

        for name in self.loss_methods:
            loss_fn = getattr(self.obj, name)
            loss = loss_fn(pred_box, target_box)
            self.assertTrue(torch.allclose(loss, torch.tensor(0.0)), f"{name}: expected 0, got {loss.item()}")


    def test_shifted_boxes(self):
        """Loss should be >0 for shifted boxes"""
        pred_box = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        target_box = torch.tensor([[1.0, 1.0, 2.0, 2.0]])
        for name in self.loss_methods:
            loss_fn = getattr(self.obj, name)
            loss = loss_fn(pred_box, target_box)
            self.assertGreater(loss, 0, f"{name}: loss should be > 0")


    def test_partial_overlap(self):
        """Loss should be >0 for shifted boxes"""
        pred_box = torch.tensor([[0.0, 0.0, 3.0, 3.0]])
        target_box = torch.tensor([[1.0, 1.0, 4.0, 4.0]])
        for name in self.loss_methods:
            loss_fn = getattr(self.obj, name)
            loss = loss_fn(pred_box, target_box)
            self.assertTrue(torch.isfinite(loss), f"{name} should be finite")
            self.assertGreater(loss, 0, f"{name}: loss should be > 0")


    def test_large_values(self):
        """Loss should remain finite value for very large coordinates"""
        pred_box = torch.tensor([[1e7, 1e7, 2e7, 3e7]])
        target_box = torch.tensor([[1e7, 1e7, 2e7, 3e7]])
        for name in self.loss_methods:
            loss_fn = getattr(self.obj, name)
            loss = loss_fn(pred_box, target_box)
            self.assertTrue(torch.isfinite(loss), f"{name} must stay finite")
            self.assertGreaterEqual(loss, 0, f"{name}: loss must be > 0")


    def test_zero_size_box(self):
        """Loss should handle degenerate boxes (zero width/height)"""
        pred_box = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
        target_box = torch.tensor([[1.0, 1.0, 2.0, 2.0]])
        for name in self.loss_methods:
            loss_fn = getattr(self.obj, name)
            loss = loss_fn(pred_box, target_box)
            self.assertTrue(torch.isfinite(loss), f"{name} devision by zero")
            self.assertGreaterEqual(loss, 0, f"{name}: loss must be > 0")

    def test_invalid_flipped_boxes(self):
        """Loss should handle boxes with reversed coodinates """
        pred_box = torch.tensor([[3.0, 3.0, 1.0, 1.0]])
        target_box = torch.tensor([[1.0, 1.0, 2.0, 2.0]])
        for name in self.loss_methods:
            loss_fn = getattr(self.obj, name)
            loss = loss_fn(pred_box, target_box)
            self.assertTrue(torch.isfinite(loss), f"{name} must handle flipped coords safely")
    
    def test_with_nans_and_inf(self):
        """Loss should handle NaN/ Inf inputs"""
        test_cases = [
            (torch.tensor([[float("nan"), 0.0, 1.0, 2.0]]), torch.tensor([[0.0, 0.0, 1.0, 2.0]])),
            (torch.tensor([[0.0, 0.0, 1.0, 2.0]]), torch.tensor([[0.0, 0.0, float("inf"), 2.0]])),
            (torch.tensor([[float("-inf"), 0.0, 1.0, 2.0]]), torch.tensor([[0.0, 0.0, 1.0, 2.0]])),
        ]
        for name in self.loss_methods:
            loss_fn = getattr(self.obj, name)
            for pred_box, target_box in test_cases:
                loss = loss_fn(pred_box, target_box)
            self.assertTrue(torch.isfinite(loss), f"{name} must handle NaN/ Inf inputs (got {loss})")


    def test_batch_processing(self):
        """Loss should work  on batches and return a scalar"""
        pred_boxes = torch.tensor([[3.0, 3.0, 1.0, 1.0], [1.0, 1.0, 3.0, 2.0]])
        target_boxes = torch.tensor([[1.0, 1.0, 2.0, 2.0], [2.0, 2.0, 3.0, 4.0]])
        for name in self.loss_methods:
            loss_fn = getattr(self.obj, name)
            loss = loss_fn(pred_boxes, target_boxes)
            self.assertEqual(loss.shape, (), f"{name}: must return a saclar")
            self.assertTrue(torch.isfinite(loss), f"{name} must stay finite")


    def test_backward(self):
        """Loss must be differentiable"""
        pred_box = torch.tensor([[3.0, 3.0, 1.0, 1.0]], requires_grad=True)
        target_box = torch.tensor([[1.0, 1.0, 2.0, 2.0]])
        for name in self.loss_methods:
            loss_fn = getattr(self.obj, name)
            loss = loss_fn(pred_box, target_box)
            loss.backward()
            self.assertIsNotNone(pred_box.grad, f"{name}: backward failed")
            self.assertTrue(torch.isfinite(pred_box.grad).all(), f"{name} gradient must be finite")     

    def test_device_and_dtype_support(self):

        "Test Loss for device and dtypes"
      
        pred_box = torch.tensor([[3.0, 3.0, 1.0, 1.0]], dtype=torch.float32)
        target_box = torch.tensor([[1.0, 1.0, 2.0, 2.0]], dtype=torch.float32)

        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        dtypes = [torch.float32, torch.float16, torch.float64]
        for name in self.loss_methods:
            loss_fn = getattr(self.obj, name)
            for device in devices:
                for dtype in dtypes:
                    b1 = pred_box.to(device=device, dtype=dtype)
                    b2 = target_box.to(device=device, dtype=dtype)
                    loss = loss_fn(b1, b2)
            self.assertTrue(torch.isfinite(loss), f"{name} failed on device {device}/{dtype}")

  

if __name__ =="__main__":
    unittest.main()


