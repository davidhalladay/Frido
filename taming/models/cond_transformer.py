import os, math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm

from main import instantiate_from_config
from taming.modules.util import SOSProvider
from taming.modules.losses.soft_cross_entropy import SoftCrossEntropy

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Net2NetTransformer(pl.LightningModule):
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 cond_stage_config,
                 permuter_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 cond_stage_key="depth",
                 downsample_cond_size=-1,
                 pkeep=1.0,
                 sos_token=0,
                 unconditional=False,
                 ):
        super().__init__()
        self.be_unconditional = unconditional
        self.sos_token = sos_token
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.init_first_stage_from_ckpt(first_stage_config)
        self.init_cond_stage_from_ckpt(cond_stage_config)
        if permuter_config is None:
            permuter_config = {"target": "taming.modules.transformer.permuter.Identity"}
        self.permuter = instantiate_from_config(config=permuter_config)
        self.transformer = instantiate_from_config(config=transformer_config)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.downsample_cond_size = downsample_cond_size
        self.pkeep = pkeep

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model

    def init_cond_stage_from_ckpt(self, config):
        if config == "__is_first_stage__":
            print("Using first stage also as cond stage.")
            self.cond_stage_model = self.first_stage_model
        elif config == "__is_unconditional__" or self.be_unconditional:
            print(f"Using no cond stage. Assuming the training is intended to be unconditional. "
                  f"Prepending {self.sos_token} as a sos token.")
            self.be_unconditional = True
            self.cond_stage_key = self.first_stage_key
            self.cond_stage_model = SOSProvider(self.sos_token)
        else:
            model = instantiate_from_config(config)
            model = model.eval()
            model.train = disabled_train
            self.cond_stage_model = model

    def forward(self, x, c):
        # one step to produce the logits
        _, z_indices = self.encode_to_z(x)
        _, c_indices = self.encode_to_c(c)


        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices

        cz_indices = torch.cat((c_indices, a_indices), dim=1)

        target = z_indices
        logits, _ = self.transformer(cz_indices[:, :-1])
        logits = logits[:, c_indices.shape[1]-1:]

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None):
        x = torch.cat((c,x),dim=1)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        if self.pkeep <= 0.0:
            assert len(x.shape)==2
            noise_shape = (x.shape[0], steps-1)
            noise = c.clone()[:,x.shape[1]-c.shape[1]:-1]
            x = torch.cat((x,noise),dim=1)
            logits, _ = self.transformer(x)
            logits = logits / temperature
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            if sample:
                shape = probs.shape
                probs = probs.reshape(shape[0]*shape[1],shape[2])
                ix = torch.multinomial(probs, num_samples=1)
                probs = probs.reshape(shape[0],shape[1],shape[2])
                ix = ix.reshape(shape[0],shape[1])
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            x = ix[:, c.shape[1]-1:]
        else:
            for k in tqdm(range(steps)):
                callback(k)
                assert x.size(1) <= block_size # make sure model can see conditioning
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed

                logits, _ = self.transformer(x_cond)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)

                probs = F.softmax(logits, dim=-1)
                if sample:
                    ix = torch.multinomial(probs, num_samples=1)
                else:
                    _, ix = torch.topk(probs, k=1, dim=-1)
                x = torch.cat((x, ix), dim=1)
            x = x[:, c.shape[1]:]
        return x

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        indices = self.permuter(indices)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_c(self, c):
        if self.downsample_cond_size > -1:
            c = F.interpolate(c, size=(self.downsample_cond_size, self.downsample_cond_size))
        quant_c, _, [_,_,indices] = self.cond_stage_model.encode(c)
        if len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)
        return quant_c, indices

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1, bhwc[3]), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, is_test=False, **kwargs):
        log = dict()

        N = None
        if lr_interface:
            x, c = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x, c = self.get_xc(batch, N)
            
        x = x.to(device=self.device)

        if self.cond_stage_key != 'caption':
            c = c.to(device=self.device)

        quant_z, z_indices = self.encode_to_z(x)
        quant_c, c_indices = self.encode_to_c(c)

        if not is_test:
            # create a "half"" sample
            z_start_indices = z_indices[:,:z_indices.shape[1]//2]
            index_sample = self.sample(z_start_indices, c_indices,
                                    steps=z_indices.shape[1]-z_start_indices.shape[1],
                                    temperature=temperature if temperature is not None else 1.0,
                                    sample=True,
                                    top_k=top_k if top_k is not None else 100,
                                    callback=callback if callback is not None else lambda k: None)
            x_sample = self.decode_to_img(index_sample, quant_z.shape)
            log["samples_half"] = x_sample

        # sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape)
        log["samples_nopix"] = x_sample_nopix

        if not is_test:
            # det sample
            z_start_indices = z_indices[:, :0]
            index_sample = self.sample(z_start_indices, c_indices,
                                    steps=z_indices.shape[1],
                                    sample=False,
                                    callback=callback if callback is not None else lambda k: None)
            x_sample_det = self.decode_to_img(index_sample, quant_z.shape)
            log["samples_det"] = x_sample_det

        # reconstruction
        x_rec = self.decode_to_img(z_indices, quant_z.shape)

        log["inputs"] = x
        log["file_name"] = self.get_img_ids(batch)
        log["reconstructions"] = x_rec

        if self.cond_stage_key in ["objects_bbox", "objects_center_points"]:
            figure_size = (x_rec.shape[2], x_rec.shape[3])
            dataset = kwargs["pl_module"].trainer.datamodule.datasets["validation"]
            label_for_category_no = dataset.get_textual_label_for_category_no
            plotter = dataset.conditional_builders[self.cond_stage_key].plot
            log["conditioning"] = torch.zeros_like(log["reconstructions"])
            for i in range(quant_c.shape[0]):
                log["conditioning"][i] = plotter(quant_c[i], label_for_category_no, figure_size)
            log["conditioning_rec"] = log["conditioning"]
        elif self.cond_stage_key != "image":
            cond_rec = self.cond_stage_model.decode(quant_c)
            if self.cond_stage_key == "segmentation":
                # get image from segmentation mask
                num_classes = cond_rec.shape[1]

                c = torch.argmax(c, dim=1, keepdim=True)
                c = F.one_hot(c, num_classes=num_classes)
                c = c.squeeze(1).permute(0, 3, 1, 2).float()
                c = self.cond_stage_model.to_rgb(c)

                cond_rec = torch.argmax(cond_rec, dim=1, keepdim=True)
                cond_rec = F.one_hot(cond_rec, num_classes=num_classes)
                cond_rec = cond_rec.squeeze(1).permute(0, 3, 1, 2).float()
                cond_rec = self.cond_stage_model.to_rgb(cond_rec)
            log["conditioning_rec"] = cond_rec
            log["conditioning"] = c
        return log

    def get_input(self, key, batch):
        x = batch[key]
        if key == 'caption':
            return x
        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        if x.dtype == torch.double:
            x = x.float()
        return x

    def get_img_ids(self, batch):
        x = batch['file_name']
        return x

    def get_xc(self, batch, N=None):
        x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        if N is not None:
            x = x[:N]
            c = c[:N]
        return x, c

    def shared_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)
        logits, target = self(x, c)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        img_ids = self.get_img_ids(batch)
        pass

    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        return optimizer


class SoftNet2NetTransformer(Net2NetTransformer):
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 cond_stage_config,
                 permuter_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 cond_stage_key="depth",
                 downsample_cond_size=-1,
                 pkeep=1.0,
                 sos_token=0,
                 unconditional=False,
                 ):
        super().__init__(
                 transformer_config,
                 first_stage_config,
                 cond_stage_config,
                 permuter_config=permuter_config,
                 ckpt_path=ckpt_path,
                 ignore_keys=ignore_keys,
                 first_stage_key=first_stage_key,
                 cond_stage_key=cond_stage_key,
                 downsample_cond_size=downsample_cond_size,
                 pkeep=pkeep,
                 sos_token=sos_token,
                 unconditional=unconditional)
                 
    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], info[2].shape[1], -1)
        indices = self.permuter(indices)
        return quant_z, indices

    def forward(self, x, c):
        # one step to produce the logits
        batch_size = x.size(0)

        # shift condition
        # c += self.first_stage_model.vocab_size

        _, z_indices = self.encode_to_z(x)
        _, c_indices = self.encode_to_c(c)

        # create embedding for soft z
        z_indices_faltten = z_indices.permute(0, 2, 1).reshape(-1, self.first_stage_model.vocab_size)
        z_embed = torch.mm(z_indices_faltten, self.transformer.tok_emb.weight)
        z_embed = z_embed.reshape(batch_size, -1, z_embed.size(-1))
        
        if self.training and self.pkeep < 1.0:
            assert self.pkeep > 1.0, 'pkeep has not been implemented'
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_embed

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        soft_target = z_indices.permute(0, 2, 1).reshape(-1, self.first_stage_model.vocab_size)
        # make the prediction
        logits, _ = self.transformer(c_indices, embeddings=z_embed[:, :-1, :], reverse_embed=True)
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        logits = logits[:, c_indices.shape[1]-1:]

        return logits, soft_target

    def shared_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)
        logits, soft_target = self(x, c)
        loss = SoftCrossEntropy(logits.reshape(-1, logits.size(-1)), soft_target, reduction='average')
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None):
        batch_size = x.size(0)
        x_indices = x
        x = c
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        if self.pkeep <= 0.0:
            raise NotImplementedError
        else:
            for k in tqdm(range(steps)):
                callback(k)
                assert x.size(1) <= block_size # make sure model can see conditioning
                x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed

                x_indices_faltten = x_indices.reshape(-1, self.first_stage_model.vocab_size)

                x_embed = torch.mm(x_indices_faltten, self.transformer.tok_emb.weight)
                x_embed = x_embed.reshape(batch_size, -1 , x_embed.size(-1))
                x_indices = x_indices.reshape(batch_size, -1, self.first_stage_model.vocab_size)


                logits, _ = self.transformer(x_cond, embeddings=x_embed, reverse_embed=True)

                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    logits = self.top_k_logits(logits, top_k)
                probs = F.softmax(logits, dim=-1).unsqueeze(1)
                x_indices = torch.cat((x_indices, probs), dim=1)

            x = x_indices
        return x

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, is_test=False, **kwargs):
        log = dict()

        N = None
        if lr_interface:
            x, c = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x, c = self.get_xc(batch, N)
        x = x.to(device=self.device)
        c = c.to(device=self.device)

        quant_z, z_indices = self.encode_to_z(x)
        quant_c, c_indices = self.encode_to_c(c)

        z_indices = z_indices.permute(0, 2, 1)

        if not is_test:
            # create a "half"" sample
            z_start_indices = z_indices[:,:z_indices.shape[1]//2, :]
            index_sample = self.sample(z_start_indices, c_indices,
                                    steps=z_indices.shape[1]-z_start_indices.shape[1],
                                    temperature=temperature if temperature is not None else 1.0,
                                    sample=True,
                                    top_k=top_k if top_k is not None else 100,
                                    callback=callback if callback is not None else lambda k: None)
            x_sample = self.decode_to_img(index_sample, quant_z.shape)
            log["samples_half"] = x_sample

        # sample
        z_start_indices = z_indices[:, :0, :]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape)
        log["samples_nopix"] = x_sample_nopix

        if not is_test:
            # det sample
            z_start_indices = z_indices[:, :0, :]
            index_sample = self.sample(z_start_indices, c_indices,
                                    steps=z_indices.shape[1],
                                    sample=False,
                                    callback=callback if callback is not None else lambda k: None)
            x_sample_det = self.decode_to_img(index_sample, quant_z.shape)
            log["samples_det"] = x_sample_det

        # reconstruction
        x_rec = self.decode_to_img(z_indices, quant_z.shape)

        log["inputs"] = x
        log["reconstructions"] = x_rec

        if self.cond_stage_key in ["objects_bbox", "objects_center_points"]:
            figure_size = (x_rec.shape[2], x_rec.shape[3])
            dataset = kwargs["pl_module"].trainer.datamodule.datasets["validation"]
            label_for_category_no = dataset.get_textual_label_for_category_no
            plotter = dataset.conditional_builders[self.cond_stage_key].plot
            log["conditioning"] = torch.zeros_like(log["reconstructions"])
            for i in range(quant_c.shape[0]):
                log["conditioning"][i] = plotter(quant_c[i], label_for_category_no, figure_size)
            log["conditioning_rec"] = log["conditioning"]
        elif self.cond_stage_key != "image":
            cond_rec = self.cond_stage_model.decode(quant_c)
            if self.cond_stage_key == "segmentation":
                # get image from segmentation mask
                num_classes = cond_rec.shape[1]

                c = torch.argmax(c, dim=1, keepdim=True)
                c = F.one_hot(c, num_classes=num_classes)
                c = c.squeeze(1).permute(0, 3, 1, 2).float()
                c = self.cond_stage_model.to_rgb(c)

                cond_rec = torch.argmax(cond_rec, dim=1, keepdim=True)
                cond_rec = F.one_hot(cond_rec, num_classes=num_classes)
                cond_rec = cond_rec.squeeze(1).permute(0, 3, 1, 2).float()
                cond_rec = self.cond_stage_model.to_rgb(cond_rec)
            log["conditioning_rec"] = cond_rec
            log["conditioning"] = c

        
        return log

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0],zshape[2],zshape[3],index.size(-1))
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1, bhwc[3]), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x


class MultiCBNet2NetTransformer(Net2NetTransformer):
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 cond_stage_config,
                 permuter_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 cond_stage_key="depth",
                 downsample_cond_size=-1,
                 pkeep=1.0,
                 sos_token=0,
                 unconditional=False,
                 ):
        super().__init__(
                 transformer_config,
                 first_stage_config,
                 cond_stage_config,
                 permuter_config=permuter_config,
                 ckpt_path=ckpt_path,
                 ignore_keys=ignore_keys,
                 first_stage_key=first_stage_key,
                 cond_stage_key=cond_stage_key,
                 downsample_cond_size=downsample_cond_size,
                 pkeep=pkeep,
                 sos_token=sos_token,
                 unconditional=unconditional)
                 
    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        if type(info[2]) is list:
            indice_list = []
            for i in info[2]:
                indice_list.append(i.reshape(len(quant_z), -1))
            indice_list = indice_list[::-1]
            indices = torch.cat(indice_list, dim=-1)
        else:
            indices = info[2][0].view(quant_z.shape[0], info[2][0].shape[1], -1)
            indices = self.permuter(indices)
        return quant_z, indices

    def forward(self, x, c):
        # one step to produce the logits
        batch_size = x.size(0)

        # shift condition
        # c += self.first_stage_model.vocab_size

        _, z_indices = self.encode_to_z(x)
        _, c_indices = self.encode_to_c(c)

        # create embedding for soft z
        # z_embed = self.transformer.tok_emb(z_indices).permute(0, 2, 1, 3).reshape(z_indices.size(0), z_indices.size(2), -1)
        if len(z_indices.shape) == 3:
            z_embed = z_indices.reshape(z_indices.shape[0], -1)
        else:
            z_embed = z_indices

        if self.training and self.pkeep < 1.0:
            assert self.pkeep > 1.0, 'pkeep has not been implemented'
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_embed

        cz_indices = torch.cat((c_indices, a_indices), dim=1)

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_embed
        # make the prediction
        logits, _ = self.transformer(cz_indices[:, :-1])
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        # for ii in range(len(logits)):
        logits = logits[:, c_indices.shape[1]-1:]
        return logits, target

    def shared_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)
        logits, target = self(x, c)
        # loss = torch.tensor(0.).cuda()
        # for logit, gt in zip(logits, target):
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        # loss = self.shared_step(batch, batch_idx)
        pass

    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, is_test=False, **kwargs):
        log = dict()

        N = None
        if lr_interface:
            x, c = self.get_xc(batch, N, diffuse=False, upsample_factor=8)
        else:
            x, c = self.get_xc(batch, N)

        x = x.to(device=self.device)
        if type(c) is torch.Tensor:
            c = c.to(device=self.device)

        quant_z, z_indices = self.encode_to_z(x) # torch.Size([4, 256, 16, 16])
        quant_c, c_indices = self.encode_to_c(c)

        # reconstruction
        x_rec = self.decode_to_img(z_indices, quant_z.shape, quant_z=quant_z)

        if not is_test:
            # create a "half"" sample
            z_start_indices = z_indices[:,:z_indices.shape[1]//2]
            index_sample = self.sample(z_start_indices, c_indices,
                                    steps=z_indices.shape[1]-z_start_indices.shape[1],
                                    temperature=temperature if temperature is not None else 1.0,
                                    sample=True,
                                    top_k=top_k if top_k is not None else 100,
                                    callback=callback if callback is not None else lambda k: None)
            # index_sample = index_sample.reshape(z_indices.shape[0], num_codebook, num_token_per_codebook).permute(0, 2, 1)
            x_sample = self.decode_to_img(index_sample, quant_z.shape)
            log["samples_half"] = x_sample
        
        # sample
        z_start_indices = z_indices[:, :0]
        index_sample = self.sample(z_start_indices, c_indices,
                                   steps=z_indices.shape[1],
                                   temperature=temperature if temperature is not None else 1.0,
                                   sample=True,
                                   top_k=top_k if top_k is not None else 100,
                                   callback=callback if callback is not None else lambda k: None)
        # index_sample = index_sample.reshape(z_indices.shape[0], num_codebook, num_token_per_codebook).permute(0, 2, 1)
        x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape)
        log["samples_nopix"] = x_sample_nopix

        if not is_test:
            # det sample
            z_start_indices = z_indices[:, :0]
            index_sample = self.sample(z_start_indices, c_indices,
                                    steps=z_indices.shape[1],
                                    sample=False,
                                    callback=callback if callback is not None else lambda k: None)
            # index_sample = index_sample.reshape(z_indices.shape[0], num_codebook, num_token_per_codebook).permute(0, 2, 1)
            x_sample_det = self.decode_to_img(index_sample, quant_z.shape)
            log["samples_det"] = x_sample_det

        log["inputs"] = x
        log["file_name"] = self.get_img_ids(batch)
        log["reconstructions"] = x_rec
        # log["teacher_force"] = x_sample_teacher
        if self.cond_stage_key in ["objects_bbox", "objects_center_points"]:
            figure_size = (x_rec.shape[2], x_rec.shape[3])
            dataset = kwargs["pl_module"].trainer.datamodule.datasets["validation"]
            label_for_category_no = dataset.get_textual_label_for_category_no
            plotter = dataset.conditional_builders[self.cond_stage_key].plot
            log["conditioning"] = torch.zeros_like(log["reconstructions"])
            for i in range(quant_c.shape[0]):
                log["conditioning"][i] = plotter(quant_c[i], label_for_category_no, figure_size)
            log["conditioning_rec"] = log["conditioning"]
        elif self.cond_stage_key != "image":
            cond_rec = self.cond_stage_model.decode(quant_c)
            if self.cond_stage_key == "segmentation":
                # get image from segmentation mask
                num_classes = cond_rec.shape[1]

                c = torch.argmax(c, dim=1, keepdim=True)
                c = F.one_hot(c, num_classes=num_classes)
                c = c.squeeze(1).permute(0, 3, 1, 2).float()
                c = self.cond_stage_model.to_rgb(c)

                cond_rec = torch.argmax(cond_rec, dim=1, keepdim=True)
                cond_rec = F.one_hot(cond_rec, num_classes=num_classes)
                cond_rec = cond_rec.squeeze(1).permute(0, 3, 1, 2).float()
                cond_rec = self.cond_stage_model.to_rgb(cond_rec)
            log["conditioning_rec"] = cond_rec
            log["conditioning"] = c

        return log

    @torch.no_grad()
    def decode_to_img(self, index, zshape, quant_z=None):
        index = self.permuter(index, reverse=True)

        start = 0
        index_group = []
        for i in range(len(self.first_stage_model.res_list)):
            index_group.append(index[:, start:int(start+self.first_stage_model.res_list[i]**2)])
            start = start + int(self.first_stage_model.res_list[i]**2)

        quant_z_group = []
        for ii, index_single in enumerate(index_group):
            res_hw = int(self.first_stage_model.res_list[ii])
            bhwc = (zshape[0],res_hw,res_hw,-1)
            quant_z_group.append(self.first_stage_model.ms_quantize[int(len(index_group)-ii-1)].get_codebook_entry(
                index_single, shape=bhwc))

        # upsample each resolutions
        for i in range(len(quant_z_group)):
            for t in range(len(quant_z_group)-i-1):
                quant_z_group[i] = self.first_stage_model.upsample(quant_z_group[i])

        quant_z_group = quant_z_group[::-1]

        quant_z = torch.cat(quant_z_group, dim=1)
        x = self.first_stage_model.decode(quant_z)
        return x

