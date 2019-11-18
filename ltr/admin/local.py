class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = 'checkpoint'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.lasot_dir = '/home/baishuai/data/LaSOTBenchmark'
        self.got10k_dir = '/home/baishuai/data/GOT-10k'
        self.trackingnet_dir = '/home/baishuai/data/dataset/TrackingNet'
        self.coco_dir = '/home/baishuai/data/datasets/mscoco2017'
        self.imagenet_dir = '/home/baishuai/data/vid/ILSVRC2016'
        self.imagenetdet_dir = '/home/baishuai/data/imagenet-det/ILSVRC'
