from pytracking.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.workspace_path = ''  # Base directory for saving network checkpoints.
    settings.network_path = settings.workspace_path + '/checkpoints/ltr'
    settings.result_plot_path = settings.workspace_path+'/results/plot'
    settings.results_path = settings.workspace_path+'/results/tracking_results'
    settings.got_packed_results_path = settings.workspace_path+'/results/GOT-10k'
    settings.tn_packed_results_path = settings.workspace_path+'/results/TrackingNet'
    settings.lst_packed_results_path = settings.workspace_path+'/results/LaSOT'
    settings.otb_path = ''
    settings.trackingnet_path = ''
    settings.lasot_path = ''
    settings.got10k_path = ''
    settings.vot_path = ''
    settings.nfs_path = ''
    settings.uav_path = ''

    settings.davis_dir = ''
    settings.got_reports_path = ''
    settings.segmentation_path = ''
    settings.tpl_path = ''
    settings.youtubevos_dir = ''

    return settings

