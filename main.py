import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset, Sampler
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only

from taming.data.utils import custom_collate

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-autoresume",
        "--autoresume",
        type=str,
        const=True,
        default=True,
        nargs="?",
        help="auto resume from logdir or checkpoint in logdir with the same name",
    )
    parser.add_argument(
        "-tb",
        "--tensorboard",
        type=str,
        const=True,
        default=False,
        nargs="?",
        help="use tensorboard",
    )
    parser.add_argument(
        "--resume_ckpt_idx",
        type=int,
        const=True,
        default=-1,
        nargs="?",
        help="auto resume from logdir or checkpoint in logdir with the same name",
    )
    parser.add_argument(
        "--save_every_n_batch",
        type=int,
        const=True,
        default=-1,
        nargs="?",
        help="save ckpt every n batch",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-log_dir", 
        "--log_dir",
        type=str,
        const=True,
        default="./",
        nargs="?",
        help="path to logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--get_codebook",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="get_codebook",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--uncond_gen_mode",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="uncond_gen_mode",
    )
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "--save_top_k",
        type=int,
        default=10,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "--split_dataset",
        type=int,
        default=1,
        help="split dataset into n groups for parallel inference",
    )
    parser.add_argument(
        "--idx_split_dataset",
        type=int,
        default=0,
        help="indicate the idx of dataset group in this run",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-tf",
        "--test_postfix",
        type=str,
        default='',
        help="post-postfix for default test dir name",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def instantiate_from_config(config, *args, **kwargs):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(*args, **config.get("params", dict()), **kwargs)


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class RandomSampler(Sampler):
    def __init__(self, data_source, num_samples=None):
        self.data_source = data_source
        self._num_samples = num_samples

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        return iter(torch.randperm(n, dtype=torch.int64)[: self.num_samples].tolist())

    def __len__(self):
        return self.num_samples

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
            print(data_cfg)
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


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config, save_ckpt_every_batch):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.save_ckpt_every_batch = save_ckpt_every_batch

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(self.config.pretty())
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(self.lightning_config.pretty())
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self.save_ckpt_every_batch != -1:
            if batch_idx % self.save_ckpt_every_batch == 0 and batch_idx != 0:
                file_name = 'epoch={:04d}_{:06d}.ckpt'.format(trainer.current_epoch, batch_idx)
                print('save model in {}'.format(os.path.join(self.ckptdir, file_name)))
                trainer.save_checkpoint(os.path.join(self.ckptdir, file_name), False)


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=False, get_codebook=False, test_postfix=''):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.get_codebook = get_codebook
        self.test_postfix = test_postfix
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            if type(images[k]) is list:
                if type(images[k][0]) is str:
                    filename = "{}_gs-{:06}_e-{:06}_b-{:06}.txt".format(
                            k,
                            global_step,
                            current_epoch,
                            batch_idx)
                    path = os.path.join(root, filename)
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    with open(path, 'w') as f:
                        f.write('\n'.join(images[k]))
            else:
                grid = torchvision.utils.make_grid(images[k], nrow=4)

                grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid*255).astype(np.uint8)
                filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                    k,
                    global_step,
                    current_epoch,
                    batch_idx)
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(grid).save(path)
    
    @rank_zero_only
    def log_local_test(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        """
        we need to log image in a batch seperately.
        """
        # TODO: organize and clean this
        root = os.path.join(save_dir, "images", split)
        os.makedirs(root, exist_ok=True)

        if 'file_name' in images:
            file_names = images.pop('file_name', None)
        else:
            try:
                file_names = ['{:06d}_{}.png'.format(batch_idx, i) for i in range(len(images[list(images)[0]]))]
            except:
                file_names = images.pop('file_name', None)
        for k in images:
            if type(images[k]) is list:
                if type(images[k][0]) is str:
                    for i in range(len(images[k])):
                        filename = "{}.txt".format(file_names[i].split('.')[0])
                        path = os.path.join(root, k, filename)
                        os.makedirs(os.path.split(path)[0], exist_ok=True)
                        with open(path, 'w') as f:
                            f.write('\n'.join([images[k][i]]))
            else:
                for i in range(len(images[k])):
                    if len(images[k].shape) == 4:
                        grid = torchvision.utils.make_grid(images[k][i].unsqueeze(0), nrow=4)
                    else:
                        grid = torchvision.utils.make_grid(images[k], nrow=4)
                        
                    grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
                    grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
                    grid = grid.numpy()
                    grid = (grid*255).astype(np.uint8)
                    filename = "{}".format(file_names[i]) ## batch size == 1
                    path = os.path.join(root, k, filename)
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    Image.fromarray(grid).save(path)
        print("batch_idx: ", batch_idx)

    def log_local_test_parallel(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx, rank):
        """
        we need to log image in a batch seperately.
        """
        if self.test_postfix != '':
            root = os.path.join(save_dir, "images", split+"_{}".format(self.test_postfix))
        else:
            root = os.path.join(save_dir, "images", split)
        os.makedirs(root, exist_ok=True)

        if 'file_name' in images:
            file_names = images.pop('file_name', None)
        else:
            try:
                file_names = ['{:06d}_{}_{}.png'.format(batch_idx, rank, i) for i in range(len(images[list(images)[0]]))]
            except:
                file_names = images.pop('file_name', None)
        for k in images:
            if type(images[k]) is list:
                if type(images[k][0]) is str:
                    for i in range(len(images[k])):
                        filename = "{}.txt".format(file_names[i].split('.')[0])
                        path = os.path.join(root, k, filename)
                        os.makedirs(os.path.split(path)[0], exist_ok=True)
                        with open(path, 'w') as f:
                            f.write('\n'.join([images[k][i]]))
            else:
                for i in range(len(images[k])):
                    if len(images[k].shape) == 4:
                        grid = torchvision.utils.make_grid(images[k][i].unsqueeze(0), nrow=4)
                    else:
                        grid = torchvision.utils.make_grid(images[k], nrow=4)
                        
                    grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
                    grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
                    grid = grid.numpy()
                    grid = (grid*255).astype(np.uint8)
                    filename = "{}".format(file_names[i]) ## batch size == 1
                    path = os.path.join(root, k, filename)
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    Image.fromarray(grid).save(path)
        print("batch_idx: ", batch_idx)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, pl_module=pl_module, is_test=(split=='test'))
            for k in images:
                if isinstance(images[k], torch.Tensor):
                    N = min(images[k].shape[0], self.max_images)
                    images[k] = images[k][:N]
                    if isinstance(images[k], torch.Tensor):
                        images[k] = images[k].detach().cpu()
                        if self.clamp:
                            images[k] = torch.clamp(images[k], -1., 1.)

            if split != 'test':
                if 'file_name' in images:
                    images.pop('file_name', None)
                if 'codebook_info' in images:
                    images.pop('codebook_info', None)
                self.log_local(pl_module.logger.save_dir, split, images,
                            pl_module.global_step, pl_module.current_epoch, batch_idx)
            else:
                if 'codebook_info' in images:
                    if self.get_codebook:
                        self.save_codebook_info(pl_module.logger.save_dir, split, images)
                    else: images.pop('codebook_info', None)
                
                if split != 'test':
                    self.log_local_test(pl_module.logger.save_dir, split, images,
                                pl_module.global_step, pl_module.current_epoch, batch_idx)
                else:
                    self.log_local_test_parallel(pl_module.logger.save_dir, split, images,
                                pl_module.global_step, pl_module.current_epoch, batch_idx, rank=pl_module.trainer.global_rank)

            if is_train:
                pl_module.train()

    def save_codebook_info(self, save_dir, split, images):
        root = os.path.join(save_dir, "codebook", split)
        os.makedirs(os.path.join(save_dir, "codebook"), exist_ok=True)
        os.makedirs(root, exist_ok=True)
        file_names = images['file_name']
        codebook_info = images.pop('codebook_info', None)
        filename = "{}".format(file_names[0]).replace('jpg', 'pt') ## batch size == 1
        path = os.path.join(root, filename)
        torch.save(codebook_info[0], path)

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # pass
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="test")



if __name__ == "__main__":

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()


    os.makedirs(os.path.join(opt.log_dir, "logs"), exist_ok=True)

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.autoresume:
        exp_list = os.listdir(os.path.join(opt.log_dir, "logs"))
        if opt.name:
            tmp_name = opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            tmp_name = os.path.splitext(cfg_fname)[0]
        exp_list = [ff for ff in exp_list if tmp_name == ff.split('-')[-1][3:]]
        exp_list = sorted(exp_list)
        if len(exp_list) > 0:
            exp_resume = os.path.join(opt.log_dir, "logs", exp_list[-1])
            if os.path.exists(os.path.join(exp_resume, 'checkpoints')):
                if len(os.listdir(os.path.join(exp_resume, 'checkpoints'))) > 0:
                    opt.resume = exp_resume
                    if 'last.ckpt' in os.listdir(os.path.join(exp_resume, 'checkpoints')):
                        opt.resume = os.path.join(exp_resume, 'checkpoints', 'last.ckpt'.format(opt.resume_ckpt_idx))
                    if opt.resume_ckpt_idx != -1:
                        opt.resume = os.path.join(exp_resume, 'checkpoints', 'epoch={:06d}.ckpt'.format(opt.resume_ckpt_idx))
                    print(f"Auto-resume checkpoint from {opt.resume}.")

    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = len(paths)-paths[::-1].index("logs")+1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            checkpoints = os.listdir(os.path.join(logdir, "checkpoints"))
            checkpoints = [ff for ff in checkpoints if 'ckpt' in ff]
            checkpoints = sorted(checkpoints)
            ckpt = os.path.join(logdir, "checkpoints", checkpoints[-1])

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs+opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[_tmp.index("logs")+1]
    else:
        if opt.name:
            name = "_"+opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_"+cfg_name
        else:
            name = ""
        nowname = now+name+opt.postfix
        logdir = os.path.join(opt.log_dir, "logs", nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        trainer_config["distributed_backend"] = "ddp"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            del trainer_config["distributed_backend"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        # NOTE wandb < 0.10.0 interferes with shutdown
        # wandb >= 0.10.0 seems to fix it but still interferes with pudb
        # debugging (wrongly sized pudb ui)
        # thus prefer testtube for now
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                }
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                    # "debug": True, 
                }
            },
            "csv": {
                "target": "pytorch_lightning.loggers.CSVLogger",
                "params": {
                    "name": "csvlogger",
                    "save_dir": logdir,
                }
            },
        }
        if opt.tensorboard:
            print("Turn on tensorboard logger.")
            default_logger_cfg = default_logger_cfgs["testtube"]
        else:
            default_logger_cfg = default_logger_cfgs["csv"]
        logger_cfg = lightning_config.logger or OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "period": 1,
                "save_top_k": -1,
                "save_last": True,
            }
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = opt.save_top_k

        modelckpt_cfg = lightning_config.modelcheckpoint or OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                    "save_ckpt_every_batch": opt.save_every_n_batch,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 1000 if opt.train else 1,
                    "max_images": 4 if opt.train else 10000,
                    "clamp": True,
                    "increase_log_steps": False,
                    "get_codebook": opt.get_codebook,
                    "test_postfix": opt.test_postfix
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    #"log_momentum": True
                }
            },
        }
        callbacks_cfg = lightning_config.callbacks or OmegaConf.create()
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)

        # data
        data = instantiate_from_config(config.data, n_split_dataset=opt.split_dataset, 
                                        idx_split_dataset=opt.idx_split_dataset)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            try:
                ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
            except:
                ngpu = 1
        else:
            ngpu = 1
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches or 1
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
        else:
            model.learning_rate = base_lr
            print("++++ NOT USING LR SCALING ++++")
            print(f"Setting learning rate to {model.learning_rate:.2e}")

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb; pudb.set_trace()

        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            # try:
            trainer.fit(model, data)
            # except Exception:
            #     melk()
            #     raise
        if not opt.no_test and not trainer.interrupted:
            print("testing time")
            if opt.uncond_gen_mode:
                print("reset seed for unconditional generation.")
                print("Set seed to {}.".format(opt.seed + trainer.local_rank))
                seed_everything(opt.seed + trainer.local_rank)
                print('Testing mode on! Auto shift random seed by number of rank.')
            trainer.test(model, datamodule=data)
    except Exception:
        if opt.debug and trainer.global_rank==0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank==0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
