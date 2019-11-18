from pytracking.evaluation import Tracker, OTBDataset, NFSDataset, UAVDataset, TPLDataset, VOTDataset, TrackingNetDataset, LaSOTDataset


def atom_nfs_uav():
    # Run three runs of ATOM on NFS and UAV datasets
    trackers = [Tracker('atom', 'default', i) for i in range(3)]

    dataset = NFSDataset() + UAVDataset()
    return trackers, dataset


def uav_test():
    # Run ATOM and ECO on the UAV dataset
    trackers = [Tracker('atom', 'default', i) for i in range(1)] + \
               [Tracker('eco', 'default', i) for i in range(1)]

    dataset = UAVDataset()
    return trackers, dataset
def atom_vot():
    trackers = [Tracker('atom', 'default_vot', i) for i in range(3)]
    dataset = VOTDataset()
    return trackers, dataset
def atom_vot2():
    trackers = [Tracker('atom', 'default_vot3', i) for i in range(3)]
    dataset = VOTDataset()
    return trackers, dataset
def drnet_vot2():
    trackers = [Tracker('drnet', 'default_vot', i) for i in range(3)]
    dataset = VOTDataset()
    return trackers, dataset
def drnet_vot_mft():
    trackers = [Tracker('drnet_mft', 'default_vot_mft', i) for i in range(3)]
    dataset = VOTDataset()
    return trackers, dataset
def drnet_vot_mft2():
    trackers = [Tracker('drnet_mft2', 'default_vot_mft', i) for i in range(3)]
    dataset = VOTDataset()
    return trackers, dataset
def eco_vot():
    trackers = [Tracker('eco', 'default', i) for i in range(3)]
    dataset = VOTDataset()
    return trackers, dataset