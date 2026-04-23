import torch
import os

dependencies = ['torch']

def balloon_detector(pretrained=True, **kwargs):
    model_path = os.path.join(os.path.dirname(__file__), 'balloon_detector_best.pt')
    model = torch.hub.load(
        'ultralytics/yolov5',
        'custom',
        path=model_path,
        force_reload=False,
        verbose=False
    )
    return model
