import argparse, os, sys, glob, datetime, yaml
from traceback import print_tb
import torch
import time
import numpy as np
from tqdm import trange
from tqdm import tqdm

from omegaconf import OmegaConf
from PIL import Image

import torchvision
from torch.utils.data import random_split, DataLoader, Dataset, Sampler
import pytorch_lightning as pl

from frido.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params
from frido.models.diffusion.ddim import DDIMSampler
from frido.models.diffusion.plms import PLMSSampler
from frido.util import instantiate_from_config_main as instantiate_from_config

from taming.data.utils import custom_collate

torch.manual_seed(23)
np.random.seed(23)

rescale = lambda x: (x + 1.) / 2.

class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, 
                 wrap=False, num_workers=None, n_split_dataset=1, idx_split_dataset=0):
        super().__init__()
        self.batch_size = batch_size
        
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap

        self.n_split_dataset = n_split_dataset
        self.idx_split_dataset = idx_split_dataset

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])
        print('Dataset statistic:')
        for k in self.datasets:
            print("number of {} data: ".format(k), len(self.datasets[k]))

    def _train_dataloader(self):
        data_loader = DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, collate_fn=custom_collate)
        return data_loader

    def _val_dataloader(self):
        data_sampler = None 
        data_loader = DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate, sampler=data_sampler)
        return data_loader

    def _test_dataloader(self):
        total_num_data = len(self.datasets["test"])
        num_data_per_group = int(len(self.datasets["test"]) / self.n_split_dataset)
        split_list = [i for i in range(0, total_num_data, num_data_per_group)] + [total_num_data]
        if len(split_list) == self.n_split_dataset + 2:
            split_list.remove(split_list[-2])
        split_list = [int(split_list[i+1] - split_list[i]) for i in range(len(split_list[:-1]))]
        assert sum(split_list) == len(self.datasets["test"]), 'missing test data after split!'
        assert len(split_list) == self.n_split_dataset, 'spliting data number error!'
        test_data_group = random_split(self.datasets["test"], split_list, generator=torch.Generator().manual_seed(42))

        return DataLoader(test_data_group[self.idx_split_dataset], batch_size=self.batch_size,
                          num_workers=self.num_workers, collate_fn=custom_collate)


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):


    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0, cond=None, 
                    unconditional_guidance_scale=1.0, 
                    unconditional_conditioning=None, plms=False
                    ):
    if plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = sampler.sample(steps, conditioning=cond, batch_size=bs, 
                shape=shape, num_stage=model.model.diffusion_model.num_stage, eta=eta, verbose=False,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning,log_every_t=20)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, cond=None, vanilla=False, 
                                custom_steps=None, eta=1.0,
                                unconditional_guidance_scale=1.0, 
                                unconditional_conditioning=None, plms=False):

    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape,
                                         make_prog_row=True)
        else:
            sample, intermediates = convsample_ddim(model,  steps=custom_steps, shape=shape,
                                                    eta=eta, cond=cond,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,plms=plms)

        t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log, intermediates

def run(model, data, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, n_samples=50000, nplog=None,
                                unconditional_guidance_scale=1.0, use_guidance=False, plms=False):
    if vanilla:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        print(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')


    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir,'*.png')))-1
    # path = logdir

    all_images = []

    if model.cond_stage_key == 'caption':
        dummy_token = 0
    elif model.cond_stage_key == 'objects_bbox' or model.cond_stage_key == 'objects':
        dummy_token = 0
    elif model.cond_stage_key == 'objects':
        dummy_token = data.datasets['test'].no_tokens - 1
    elif model.cond_stage_key == 'class_label':
        dummy_token = 0
    else:
        raise ValueError('The cond_stage_key in model {} is not support.'.format(model.cond_stage_key))
    
    dataloader = data.test_dataloader()
    print(f"Running conditional sampling for {len(dataloader)*batch_size} samples")
    for idx, batch in enumerate(tqdm(dataloader)):
        z, c, x, xrec, xc = model.get_input(batch, model.first_stage_key,
                                        return_first_stage_outputs=True,
                                        force_c_encode=True,
                                        return_original_cond=True,
                                        bs=None)
        if use_guidance:
            if model.cond_stage_key == 'class_label':
                unconditional_conditioning = torch.full_like(c, dummy_token)

            elif model.cond_stage_key == 'objects_bbox':
                unconditional_conditioning = torch.full_like(c, dummy_token)
            else:
                if model.cond_stage_model.use_tknz_fn == True:
                    xc_tmp = ['' for i in range(len(xc))]
                else:
                    unconditional_conditioning = torch.full_like(xc[model.cond_stage_key], dummy_token)
                    xc_tmp = xc.copy()
                    xc_tmp[model.cond_stage_key] = unconditional_conditioning
                unconditional_conditioning = model.get_learned_conditioning(xc_tmp)
        else:
            unconditional_conditioning = None
        logs, intermediates = make_convolutional_sample(model, batch_size=len(z), cond=c,
                                            vanilla=vanilla, custom_steps=custom_steps,
                                            eta=eta, unconditional_guidance_scale=unconditional_guidance_scale, 
                                            unconditional_conditioning=unconditional_conditioning, plms=plms)

        try:
            logs["file_name"] = model.get_img_ids(batch)
        except:
            pass

        logs["inputs"] = x
        logs["reconstruction"] = xrec
        if model.model.conditioning_key is not None:
            if hasattr(model.cond_stage_model, "decode"):
                xc = model.cond_stage_model.decode(c)
                logs["conditioning"] = xc
            elif model.cond_stage_key in ["caption"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
                logs["conditioning"] = xc
            elif model.cond_stage_key == 'objects':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["objects"].cpu().tolist())
                logs['conditioning'] = xc
            elif model.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                logs['conditioning'] = xc
            elif model.cond_stage_key == "objects_bbox":
                figure_size = (x.shape[2], x.shape[3])
                dataset = data.datasets["validation"]
                label_for_category_no = dataset.get_textual_label_for_category_no
                plotter = dataset.conditional_builders[model.cond_stage_key].plot
                logs["conditioning"] = torch.zeros_like(logs["reconstruction"])
                for i in range(logs["conditioning"].shape[0]):
                    logs["conditioning"][i] = plotter(xc[i].long(), label_for_category_no, figure_size)
            elif isimage(xc):
                logs["conditioning"] = xc
                
        n_saved = save_logs(logs, logdir, n_saved=n_saved, keys=["sample", "inputs", "conditioning"])

        all_images.extend([custom_to_np(logs["sample"])])

    all_img = np.concatenate(all_images, axis=0)
    all_img = all_img[:n_samples]
    shape_str = "x".join([str(x) for x in all_img.shape])
    nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
    np.savez(nppath, all_img)

    print(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(logs, path, n_saved=0, keys=["sample"], np_path=None):
    for k in logs:
        if k in keys:
            key = k
            os.makedirs(os.path.join(path, key), exist_ok=True)
            batch = logs[key]
            if np_path is None:
                if len(batch.shape) == 3:
                    filename = f"{key}.png"
                    img = custom_to_pil(batch)
                    imgpath = os.path.join(path, key, filename)
                    img.save(imgpath)
                    continue
                for idx, x in enumerate(batch):
                    if "file_name" in logs:
                        filename = "{}.png".format(logs['file_name'][idx].split('.')[0])
                    else:
                        filename = f"{key}_{n_saved:06}.png"
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, key, filename)
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-cfg",
        "--cfg_path",
        type=str,
        nargs="?",
        help="path to config.",
    )
    parser.add_argument(
        "-name",
        "--exp_name",
        type=str,
        default="v0",
        help="path to config.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default='',
        help="path to output.",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=-1
    )
    parser.add_argument(
        "-plms",
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=200
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    parser.add_argument(
        "-G",
        "--use_guidance",
        action='store_true',
        help="use classifier-free guidance",
        default=False
    )
    parser.add_argument(
        "-gs",
        "--guidance_scale",
        type=float,
        nargs="?",
        help="guidance_scale",
        default=1.
    )
    parser.add_argument(
        "-ngpu",
        "--num_gpus",
        type=int,
        default=1,
        help="split dataset into n groups for parallel inference",
    )
    parser.add_argument(
        "-igpu",
        "--gpu_idx",
        type=int,
        default=0,
        help="indicate the idx of dataset group in this run",
    )
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
    else:
        pl_sd = {"state_dict": None}
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None

    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        # paths = opt.resume.split("/")
        try:
            logdir = '/'.join(opt.resume.split('/')[:-1])
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt_filename = sorted(os.listdir(os.path.join(logdir, 'checkpoints')))[-1]
        ckpt = os.path.join(logdir, 'checkpoints', ckpt_filename)

    base_configs = [opt.cfg_path] #sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True

    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "": locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)

    model = load_model(config, ckpt, gpu, eval_mode)
    print(75 * "=")
    print("logging to:")
    if opt.output_path != '':
        logdir = opt.output_path
    else:
        logdir = os.path.join(logdir, "samples", opt.exp_name)
    imglogdir = os.path.join(logdir, "img")
    numpylogdir = os.path.join(logdir, "numpy")

    os.makedirs(imglogdir, exist_ok=True)
    os.makedirs(numpylogdir, exist_ok=True)
    print(logdir)
    print(75 * "=")

    # create dataset
    # data
    data = None
    
    if opt.n_samples != -1:
        config.data['params']['validation']['params']['num_sample'] = opt.n_samples
        config.data['params']['test']['params']['num_sample'] = opt.n_samples

    if model.cond_stage_model is not None:
        data = instantiate_from_config(config.data, n_split_dataset=opt.num_gpus, 
                                            idx_split_dataset=opt.gpu_idx)
        data.prepare_data()
        data.setup()
        opt.batch_size = data.batch_size

    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = vars(opt)

    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    print(sampling_conf)

    run(model, data, imglogdir, eta=opt.eta,
        vanilla=opt.vanilla_sample, custom_steps=opt.custom_steps,
        batch_size=opt.batch_size, nplog=numpylogdir, use_guidance=opt.use_guidance, 
        unconditional_guidance_scale=opt.guidance_scale, plms=opt.plms)

    print("done.")