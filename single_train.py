import argparse
import logging
import os
import os.path as osp
import torch
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           scandir, tensor2img)

from basicsr.utils.options import copy_opt_file, dict2str
from omegaconf import OmegaConf
import time
from ldm.data.dataset_ps_keypose import PsKeyposeDataset
from basicsr.utils.dist_util import get_dist_info, init_dist, master_only
from ldm.modules.encoders.adapter import Adapter
from ldm.util import load_model_from_config

from ldm.inference_base import (train_inference, diffusion_inference, get_adapters, get_base_argument_parser,
                                get_sd_models)
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import (ExtraCondition, get_adapter_feature, get_cond_openpose)


@master_only
def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    os.makedirs(path, exist_ok=True)
    os.makedirs(osp.join(path, 'models'))
    os.makedirs(osp.join(path, 'training_states'))
    os.makedirs(osp.join(path, 'visualization'))

def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
        
def str2int(v: str) -> int:
    if isinstance(v, int):
        return v
    if '*' in v:
        assert isinstance(v, str), 'unrecognized / illegal keyword: \"max_resolution\"'
        v = v.replace(' ', '')
        l = v.split('*')
        re = 1
        for x in l:
            re *= int(x)
        return re
    else:
        raise RuntimeError('Unkonwn Exception.')
        

def load_resume_state(opt):
    resume_state_path = None
    if opt.auto_resume:
        state_path = osp.join('experiments', opt.name, 'training_states')
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt.resume_state_path = resume_state_path

    if resume_state_path is None:
        resume_state = None
    else:
        device_id = torch.cuda.current_device()
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id))
    return resume_state


def parsr_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bsize",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="twice of the amount of GPU"
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--auto_resume",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--sd_ckpt",
        type=str,
        default="models/v1-5-pruned.ckpt",
        help="path to checkpoint of sd model",
    )
    parser.add_argument(
        "--adapter_ori",
        type=str,
        default="models/t2iadapter_keypose_sd14v1.pth"
    )
    parser.add_argument(
        "--adapter_ckpt",
        type=str,
        default="models/t2iadapter_keypose_sd14v1.pth"
    )
    parser.add_argument(
        "--cond_weight",
        type=str,
        default="models/t2iadapter_keypose_sd14v1.pth"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/sd-v1-train.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="single_ps_keypose",
        help="experiment name",
    )
    parser.add_argument(
        "--print_fq",
        type=int,
        default=100,
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        '--cond_tau',
        type=float,
        default=1.0,
        help='timestamp parameter that determines until which step the adapter is applied, '
        'similar as Prompt-to-Prompt tau'
    )
    parser.add_argument(
        '--launcher',
        default='pytorch',
        type=str,
        help='node rank for distributed training'
    )
    parser.add_argument(
        '--caption_path',
        default='Datasets/Captions/captions.csv',
        type=str,
        help='path for captions'
    )
    parser.add_argument(
        '--keypose_folder',
        default='Datasets/Keypose/',
        type=str,
        help='path for keypose'
    )
    parser.add_argument(
        '--data_size',
        default=500,
        type=int,
        help='the amount of the data chosen from Datasets'
    )
    parser.add_argument(
        '--vae_ckpt',
        default=None,
        type=str,
        help='vae checkpoint, anime SD models usually have seperate vae ckpt that need to be loaded'
    )
    parser.add_argument(
        '--device',
        default="cuda",
        type=str,
        help='device name'
    )
    parser.add_argument(
        '--sampler',
        type=str,
        default='ddim',
        choices=['ddim', 'plms'],
        help='sampling algorithm, currently, only ddim and plms are supported, more are on the way',
    )
    parser.add_argument(
        '--resize',
        type=str2bool,
        default=False,
        help='resize image shape'
    )
    parser.add_argument(
        "--inter",
        type=str,
        default='inter_cubic',
        choices=['inter_cubic', 'inter_liinear', 'inter_nearest', 'inter_lanczos4'],
        help='resize shape'
    )
    parser.add_argument(
        "--factor",
        type=int,
        default=8,
        help='download sample factor'
    )
    parser.add_argument(
        "--max_resolution",
        type=str2int,
        default=512 * 512,
        help='quality of generated image'
    )


    opt = parser.parse_args()
    return opt


def rates(ratios: dict):
    assert 'alphas' in ratios, 'Invalid ratios.'
    alphas = ratios['alphas']
    return (alphas / (1. - alphas)).sum(dim=0, keepdim=False)





def main():
    opt = parsr_args()
    print('loading configs...')
    config = OmegaConf.load(f"{opt.config}")
    # print(opt.launcher)
    # init_dist(opt.launcher)
    torch.backends.cudnn.benchmark = True
    device = 'cuda'
    # torch.cuda.set_device(opt.local_rank)

    print('reading datasets...')
    train_dataset = PsKeyposeDataset(opt.caption_path, opt.keypose_folder, resize=opt.resize,\
                                     interpolation=opt.inter, factor=opt.factor, max_resolution=opt.max_resolution)
    opt.H, opt.W = train_dataset.item_shape
    # downloaded: H, W
    setattr(opt, 'resize_short_edge', None)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.bsize,
        shuffle=True)

    # Stable-Diffusion Model
    model, sampler = get_sd_models(opt)

    print('loading adapters from {0}'.format(opt.adapter_ori))

    primary_adapter = get_adapters(opt, getattr(ExtraCondition, "openpose"))
    secondary_adapter = get_adapters(opt, getattr(ExtraCondition, "openpose"))
    # Adapter(cin=3 * 64, channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False).to(device)
    # hyper-parameters remained to be adjust


    params = list(secondary_adapter['model'].parameters())
    optimizer = torch.optim.AdamW(params, lr=config['training']['lr'])

    experiments_root = osp.join('experiments', opt.name)

    # resume state
    print('getting resume state...')

    resume_state = load_resume_state(opt)
    if resume_state is None:
        mkdir_and_rename(experiments_root)
        start_epoch = 0
        current_iter = 0
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{opt.name}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        logger.info(dict2str(config))
    else:
        # WARNING: should not use get_root_logger in the above codes, including the called functions
        # Otherwise the logger will not be properly initialized
        log_file = osp.join(experiments_root, f"train_{opt.name}_{get_time_str()}.log")
        logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
        logger.info(get_env_info())
        logger.info(dict2str(config))
        resume_optimizers = resume_state['optimizers']
        optimizer.load_state_dict(resume_optimizers)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, " f"iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']

    # copy the yml file to the experiment root
    copy_opt_file(opt.config, experiments_root)

    # training
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    model_reflect = lambda x: model.get_first_stage_encoding(
        model.encode_first_stage((data[x] * 2 - 1).to(device))).type(torch.float32)
    for epoch in range(start_epoch, opt.epochs):
        # train_dataloader.sampler.set_epoch(epoch)
        epoch_start_time = time.time()
        # train
        for _, data in enumerate(train_dataloader):
            current_iter += 1
            with torch.no_grad():
                c = model.get_learned_conditioning(data['prompt'])
                # CLIP
                
                # A_0 = tensor2img(model_reflect('primary'))
                B_0 = tensor2img(model_reflect('secondary'))
                
                const_B = get_cond_openpose(opt, B_0, cond_inp_type='openpose')  # only need openpose
                print('data[...].shape = ', data['secondary'].shape)
                print('B_0.shape = ', B_0.shape)
                print('const_B.shape = ', const_B.shape)

                # assert const_B.shape[0] == opt.H * opt.factor or const_B.shape[1] == opt.W * opt.factor, "op-wh = ({0}, {1})".format(opt.H, opt.W)

                features_A  = primary_adapter['model'](data['primary'].to(device))

                # already went through 'img2tensor'
                
                samples_A, _ = train_inference(opt, c, model, sampler, features_A, get_cond_openpose)

            optimizer.zero_grad()
            model.zero_grad()
            primary_adapter.zero_grad()

            # features_B, append_B = secondary_adapter(data['secondary'].to(device))
            features_B = secondary_adapter(data['secondary'].to(device))
            samples_B, ratios = train_inference(opt, model, sampler, features_B, get_cond_openpose)

            u = (samples_B - const_B) ** 2
            v = (samples_B - samples_A) ** 2
            Expectation = 2 * rates(ratios) * u.sum() + v.sum()

            loss_dict = {}
            log_prefix = 'Ps-Adapter-single-train'
            loss_dict.update({f'{log_prefix}/loss_u': u})
            loss_dict.update({f'{log_prefix}/loss_v': v})
            loss_dict.update({f'{log_prefix}/loss_Expectation': Expectation})

            print("[%5d|%5d] %.2f(s) Exception Loss: %.6f " % (epoch, time.time() - epoch_start_time, opt.epochs-start_epoch+1, Expectation))

            Expectation.backward()
            optimizer.step()

            if (current_iter + 1) % opt.print_fq == 0:
                logger.info(loss_dict)

                # save checkpoint
                rank, _ = get_dist_info()
                if (rank == 0) and ((current_iter + 1) % config['training']['save_freq'] == 0):
                    save_filename = f'model_ad_{current_iter + 1}.pth'
                    save_path = os.path.join(experiments_root, 'models', save_filename)
                    save_dict = {}
                    state_dict = secondary_adapter.state_dict()
                    for key, param in state_dict.items():
                        if key.startswith('module.'):  # remove unnecessary 'module.'
                            key = key[7:]
                        save_dict[key] = param.cpu()
                    torch.save(save_dict, save_path)
                    # save state
                    state = {'epoch': epoch, 'iter': current_iter + 1, 'optimizers': optimizer.state_dict()}
                    save_filename = f'{current_iter + 1}.state'
                    save_path = os.path.join(experiments_root, 'training_states', save_filename)
                    torch.save(state, save_path)


if __name__ == "__main__":
    main()
