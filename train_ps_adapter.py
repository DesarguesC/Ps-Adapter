# /usr/local/nccl_2.17.1-1+cuda12.0_x86_64/lib
import argparse
import logging
import os
import os.path as osp
import torch
import numpy as np
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           scandir, tensor2img)

from basicsr.utils.options import copy_opt_file, dict2str
from omegaconf import OmegaConf
import time
from ldm.modules.extra_condition.openpose.api import OpenposeInference
from ldm.data.dataset_ps_keypose import PsKeyposeDataset
from basicsr.utils.dist_util import get_dist_info, master_only, init_dist
from ldm.modules.encoders.adapter import Adapter
from ldm.util import load_model_from_config
from ldm.data.dataset_ps_keypose import deal, Inter
from ldm.inference_base import (train_inference, diffusion_inference, get_adapters, get_base_argument_parser,
                                get_sd_models)
import cv2
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import (ExtraCondition, get_adapter_feature, get_cond_openpose)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


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
        default=4,
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
        default="train_depth",
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
        "--gpus",
        default=[0, 1, 2, 3],
        help="gpu idx",
    )
    parser.add_argument(
        '--local_rank',
        default=0,
        type=int,
        help='node rank for distributed training'
    )
    
    # read form environment      ?
    
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


    opt = parser.parse_args()
    return opt

def rates(ratios: dict):
    assert 'alphas' in ratios, 'Invalid ratios.'
    alphas = ratios['alphas']
    return (alphas / (1. - alphas)).sum(dim=0, keepdim=False)

def resize_tensor_image(A, B, inter):
    # resize A as B
    h, w, _ = B.shape
    A_ = (A.cpu().numpy()).astype(np.float32)
    A = cv2.resize(A_, (h,w), interpolation=Inter[inter])
    A = torch.from_numpy(A)
    A, B = deal(A), deal(B)
    assert A.shape == B.shape, f'Resize Failed: A.shape = {A.shape}, B.shape = {B.shape}'
    return A.cpu(), B.cpu()





def main():
    opt = parsr_args()
    print('loading configs...')
    config = OmegaConf.load(f"{opt.config}")
    
    
    # x = opt.gpus
    # print(x)
    # exit(0)
    
    
    torch.cuda.set_device(opt.local_rank)
    print('start init')
    torch.distributed.init_process_group(backend='NCCL')
    
    print('init ends')
    
    # init_dist(opt.launcher)
    torch.backends.cudnn.benchmark = True
    device = 'cuda'
    torch.cuda.set_device(opt.local_rank)
    

    print('reading datasets...')
    train_dataset = PsKeyposeDataset(opt.caption_path, opt.keypose_folder, interpolation='inter_lanczos', resize=opt.resize, factor=opt.factor)
    opt.H, opt.W = train_dataset.item_shape
    print('base shape: ', (opt.H, opt.W))
    max_resolution = opt.W * opt.H
    setattr(opt, 'max_resolution', max_resolution)
    setattr(opt, 'resize_short_edge', None)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.bsize,
        shuffle= (train_sampler is None),
        num_workers=opt.num_workers,
        pin_memory=True,
        sampler=train_sampler)

    # Stable-Diffusion Model
    
    print('loading stable-diffusion model from {0}'.format(opt.sd_ckpt))

    # sd model
    model, sampler = get_sd_models(opt)
    
    print('loading adapters from {0}'.format(opt.adapter_ori))
    
    primary_adapter = get_adapters(opt, getattr(ExtraCondition, "openpose"))
    secondary_adapter = get_adapters(opt, getattr(ExtraCondition, "openpose"))
    # Adapter(cin=3 * 64, channels=[320, 640, 1280, 1280][:4], nums_rb=2, ksize=1, sk=True, use_conv=False).to(device)
    # hyper-parameters remained to be adjust

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[opt.local_rank],
        output_device=opt.local_rank)
    primary_adapter = torch.nn.parallel.DistributedDataParallel(
        primary_adapter['model'],
        device_ids=[opt.local_rank],
        output_device=opt.local_rank)
    secondary_adapter = torch.nn.parallel.DistributedDataParallel(
        secondary_adapter['model'],
        device_ids=[opt.local_rank],
        output_device=opt.local_rank)

    params = list(secondary_adapter.module.parameters())
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
    model_reflect = lambda x: model.module.get_first_stage_encoding(    # ???
        model.module.encode_first_stage((data[x] * 2 - 1).to(device))).type(torch.float32)
    cond_model = OpenposeInference().to(device)
    
    for epoch in range(start_epoch, opt.epochs):
        
        ss = 0
        train_dataloader.sampler.set_epoch(epoch)
       
        # train
        for _, data in enumerate(train_dataloader):
            current_iter += 1
            epoch_start_time = time.time()
            ss += 1
            with torch.no_grad():
                # CLIP
                c = model.module.get_learned_conditioning(data['prompt'])
                
                B_0 = tensor2img(model_reflect('secondary'))
                const_B = get_cond_openpose(opt, B_0, cond_inp_type='openpose')
                # already went through 'img2tensor'
                features_A = primary_adapter.module(data['primary'].to(device))
                samples_A, _ = train_inference(opt, c, model.module, sampler, features_A, cond_model=cond_model, loss_mode=True)

            optimizer.zero_grad()
            model.zero_grad()
            primary_adapter.zero_grad()
            features_B= secondary_adapter.module(data['secondary'].to(device))
            samples_B, ratios = train_inference(opt, c, model.module, sampler, features_B, cond_model=cond_model, loss_mode=True)

            assert len(samples_B) == len(samples_A), 'qwq'           

            const_B = const_B.to(torch.float32)
            
            print('Training Base Info: ')
            print(f'latent shape: {samples_B[0].shape}, const shape: {const_B.shape}')
            
            sh = torch.from_numpy(samples_A[0].astype(np.float32)).shape
            u, v = torch.zeros(sh, dtype=torch.float32, requires_grad=True), \
               torch.zeros(sh, dtype=torch.float32, requires_grad=True)
            
            for i in range(len(samples_A)):
                
                B = torch.from_numpy(np.float32(samples_B[i])).squeeze()
                A = torch.from_numpy(np.float32(samples_A[i])).squeeze()
                const_B = const_B.squeeze()
                assert len(A.shape)==3
                assert A.shape == B.shape
                
                A, B, const_B = deal(A), deal(B), deal(const_B)
                
                const_B, B = resize_tensor_image(const_B, B, inter=opt.inter)
                u = u + (B - const_B) ** 2
                v = v + (B - A) ** 2
            u, v = u.sum(), v.sum()
            
            print((A-B).sum())
            
            Expectation_Loss = 2 * rates(ratios) * u.sum() + v.sum()
            Expectation_Loss.backward()
            optimizer.step()

            loss_dict = {}
            log_prefix = 'Ps-Adapter-multiGPUs-train'
            loss_dict.update({f'{log_prefix}/loss_u': u})
            loss_dict.update({f'{log_prefix}/loss_v': v})
            loss_dict.update({f'{log_prefix}/Expectation_Loss': Expectation_Loss})

            print("[%4d|%4d] IN %2d-th DATA, TIME: %.2f(s),  U: %.6f, V: %.6f, Exception Loss: %.6f " % \
             (epoch+1, opt.epochs-start_epoch, ss, time.time() - epoch_start_time, u, v, Expectation_Loss))



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
