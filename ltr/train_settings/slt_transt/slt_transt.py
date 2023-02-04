import torch

from ltr.dataset import Lasot, Got10k, TrackingNet
from ltr.data import sequence_sampler, SLTLoader
import ltr.models.tracking.transt as transt_models
from ltr.actors.slt_transt_actor import SLTTransTActor
from ltr.trainers.slt_transt_trainer import SLTTransTTrainer
from ltr import MultiGPU
from ltr.admin import loading

from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone


def run(settings):
    # Most common settings are assigned in the settings struct
    settings.device = 'cuda'
    settings.description = 'SLT-TransT with default settings.'
    settings.num_workers = 4
    settings.multi_gpu = True
    settings.print_interval = 1

    # SLT settings
    settings.num_epoch = 120
    settings.num_per_epoch = 1000
    settings.num_seq = 8
    settings.num_seq_backward = 2
    settings.num_frames = 24
    settings.slt_loss_weight = 15.0
    settings.clip_grad = True
    settings.grad_max_norm = 100.0

    # Transformer
    settings.position_embedding = 'sine'
    settings.hidden_dim = 256
    settings.dropout = 0.1
    settings.nheads = 8
    settings.dim_feedforward = 2048
    settings.featurefusion_layers = 4

    # Train datasets
    lasot_train = Lasot(settings.env.lasot_dir, split='train')
    got10k_train = Got10k(settings.env.got10k_dir, split='vottrain')
    trackingnet_train = TrackingNet(settings.env.trackingnet_dir, set_ids=list(range(4)))

    # Train sampler and loader
    settings.max_gap = 300
    settings.max_interval = 10
    settings.interval_prob = 0.3
    dataset_train = sequence_sampler.SequenceSampler([lasot_train, got10k_train, trackingnet_train], [1,1,1],
                                        samples_per_epoch=settings.num_per_epoch, max_gap=settings.max_gap, max_interval=settings.max_interval,
                                                     num_search_frames=settings.num_frames, num_template_frames=1,
                                                     frame_sample_mode='random_interval', prob=settings.interval_prob)

    loader_train = SLTLoader('train', dataset_train, training=True, batch_size=settings.num_seq, num_workers=settings.num_workers,
                             shuffle=False, drop_last=True)

    # Create network and actor
    model = transt_models.transt_resnet50(settings)

    # Wrap the network for multi GPU training
    if settings.multi_gpu:
        model = MultiGPU(model, dim=0)

    objective = transt_models.slt_transt_loss(settings)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # Actor with tracker params
    params = TrackerParams()
    params.use_gpu = True
    params.no_neg_logit = False
    params.sig_eps = 0.01
    params.temp = 4.0
    params.net = NetWithBackbone(net_path='transt.pth',
                                 use_gpu=params.use_gpu)
    actor = SLTTransTActor(net=model, objective=objective, params=params)

    # Optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": 1e-6,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=1e-5,
                                  weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100)

    # Create trainer
    trainer = SLTTransTTrainer(actor, [loader_train], optimizer, settings, lr_scheduler)

    # Load pretrained model
    net = actor.net.module if settings.multi_gpu else actor.net
    checkpoint_dict = loading.torch_load_legacy('{$PATH_TO_YOUR_MODELS}/transt.pth')
    net.load_state_dict(checkpoint_dict['net'])

    # Run training (set fail_safe=False if you are debugging)
    trainer.train(settings.num_epoch, load_latest=True, fail_safe=True)

