from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    pytracking_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    print(pytracking_path)
    settings = EnvSettings()

    # Set your local paths here.

    settings.got10k_path = ''
    settings.lasot_path = ''
    settings.network_path = pytracking_path+'/networks/'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.results_path = ''    # Where to store tracking results
    settings.tpl_path = ''
    settings.trackingnet_path = ''
    settings.uav_path = ''
    settings.vot_path = ''

    return settings

