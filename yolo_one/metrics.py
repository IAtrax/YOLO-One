"""
IATrax Team - 2023 - https://iatrax.com

LICENSE: MIT

METRICS MODULE FOR YOLO-ONE

"""

class SimpleYoloOneMetrics:
    """Simplified metrics for debugging"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.reset()
    
    def reset(self):
        self.total_predictions = 0
        self.total_targets = 0
    
    def update(self, predictions, targets, **kwargs):
        """Simplified update without heavy computation"""
        
        # Count only, no processing
        if isinstance(predictions, dict):
            pred_tensor = predictions['detections'][0]
        else:
            pred_tensor = predictions[0]
        
        batch_size = pred_tensor.shape[0]
        self.total_predictions += batch_size
        self.total_targets += len(targets)
        
        print(f"📊 Batch processed: {batch_size} predictions, {len(targets)} targets")
    
    def compute(self):
        """Return simple metrics"""
        return {
            'total_predictions': self.total_predictions,
            'total_targets': self.total_targets,
            'mAP': 0.5,  # Dummy value
            'avg_loss': 0.1  # Dummy value
        }