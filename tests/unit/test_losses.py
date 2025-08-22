import unittest
import torch


from yolo_one.losses import YoloOneLoss


class TestBoxLoss(unittest.TestCase):

    def setUp(self):

        """List of losses to test"""
        self.obj = YoloOneLoss()
        self.loss_methods = ["_ciou_loss", "_eiou_loss", "_meiou_loss", "_siou_loss"]
  

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


