from ast import Raise

from yaml import ScalarEvent
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from frido.util import instantiate_from_config_main as instantiate_from_config

from taming.modules.diffusionmodules.model import Encoder, Decoder, MSEncoder
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from taming.modules.vqvae.quantize import GumbelQuantize
from taming.modules.vqvae.quantize import EMAVectorQuantizer

class MSFPNVQModel(pl.LightningModule):
    def __init__(self,
                 edconfig,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 fusion='concat',
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 on_vit = [],
                 use_aux_loss=False, 
                 unsample_type='nearest',
                 quant_beta=0.25,
                 legacy=True, 
                 init_normal=False, 
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = MSEncoder(**edconfig)
        self.decoder = Decoder(**ddconfig)
        
        self.loss = instantiate_from_config(lossconfig)
        self.fusion = fusion

        self.embed_dim = embed_dim
        self.use_aux_loss = use_aux_loss
        self.unsample_type = unsample_type

        assert len(n_embed) == edconfig['multiscale'], 'multiscale mode. dim of n_embed is incorrect.'
        assert len(n_embed) == len(embed_dim), 'multiscale mode. dim of n_embed is incorrect.'

        self.res_list = []
        for i in range(self.encoder.multiscale):
            self.res_list.append(self.encoder.resolution / 2**(self.encoder.num_resolutions - i - 1))

        self.ms_quantize = nn.ModuleList()
        self.ms_quant_conv = nn.ModuleList()
        if self.fusion == 'concat':
            for i in range(len(n_embed)):
                self.ms_quantize.append(VectorQuantizer(n_embed[i], embed_dim[i], beta=quant_beta,
                                                remap=remap, sane_index_shape=sane_index_shape, legacy=legacy, init_normal=init_normal)
                                        )
                in_channel = 2*edconfig["z_channels"][i] if edconfig["double_z"] else edconfig["z_channels"][i]
                self.ms_quant_conv.append(torch.nn.Conv2d(in_channel, embed_dim[i], 1))
            embed_dim_sum = sum(embed_dim)
        else:
            self.ms_quantize.append(VectorQuantizer(n_embed[0], embed_dim[0], beta=quant_beta,
                                                remap=remap, sane_index_shape=sane_index_shape, legacy=legacy, init_normal=init_normal)
                                        )
            in_channel = 2 * edconfig["z_channels"][0] if edconfig["double_z"] else edconfig["z_channels"][0]
            self.ms_quant_conv.append(torch.nn.Conv2d(in_channel, embed_dim[0], 1))

            embed_dim_sum = embed_dim[0]
        self.post_quant_conv = torch.nn.Conv2d(embed_dim_sum, ddconfig["z_channels"], 1)

        # share structure
        self.upsample = nn.ModuleList()
        self.shared_decoder = nn.ModuleList()
        self.shared_post_quant_conv = nn.ModuleList()
        for i in range(len(n_embed)-1):
            self.upsample.append(nn.ConvTranspose2d(
                embed_dim[0], embed_dim[0], 4, stride=2, padding=1
            ))
            self.shared_post_quant_conv.append(torch.nn.Conv2d(embed_dim[0], edconfig["z_channels"][0], 1))
            self.shared_decoder.append(Decoder(double_z=False, z_channels=sum(embed_dim[:(i+2)]), resolution=256, in_channels=embed_dim[:(i+2)], 
                     out_ch=embed_dim[0], ch=128, ch_mult=[ 1 ], num_res_blocks=2, attn_resolutions=[2, 4, 8, 16, 32, 64], dropout=0.0))

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        missing, unexpected = self.load_state_dict(sd, strict=False)

        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")
        

    def encode(self, x):
        h_ms = self.encoder(x)

        qaunt_ms = []
        emb_loss_ms = []
        info_ms = [[], [], []]
        h_ms = h_ms[::-1]
        prev_h = []
        for ii in range(len(h_ms)):

            if len(prev_h) != 0:
                prev_h_new = []
                for j in range(ii):
                    prev_h[j] = self.upsample[ii-1](prev_h[j])
                    prev_h_j = self.shared_post_quant_conv[ii-1](prev_h[j]) #should not replace
                    prev_h_new.append(prev_h_j)
                
                quant = torch.cat((*prev_h[:ii], h_ms[ii]), dim=1)
                quant = self.shared_decoder[ii-1](quant)
                # quant = quant + prev_h
            else:
                quant = h_ms[ii]

            h = self.ms_quant_conv[ii](quant)
            quant, emb_loss, info = self.ms_quantize[ii](h)

            qaunt_ms.append(quant)
            emb_loss_ms.append(emb_loss)
            for jj in range(len(info)):
                info_ms[jj].append(info[jj])
            prev_h.append(quant)

        qaunt_ms = qaunt_ms[::-1]
        # # upsample each resolutions
        for i in range(len(h_ms)):
            for t in range(i):
                qaunt_ms[i] = F.interpolate(qaunt_ms[i], scale_factor=2)

        quant = torch.cat(qaunt_ms, dim=1) # channel-wise concate
        emb_loss = sum(emb_loss_ms)
        return quant, emb_loss, info_ms

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_info=False):
        quant, diff, info = self.encode(input)
        quant_aux = quant.clone()

        # TODO: remove dummy
        quant_aux[:, :-1*self.embed_dim[-1], :, :] = 0
        quant_aux2 = quant.clone()
        quant_aux2[:, self.embed_dim[-1]:, :, :] = 0
        
        dec = self.decode(quant)
        dec_aux = self.decode(quant_aux)
        dec_aux2 = self.decode(quant_aux2)
        
        dec_aux = [dec_aux, dec_aux2]
        
        if self.use_aux_loss:
            return dec, dec_aux, diff, info
        
        if return_info:
            return dec, diff, info
        return dec, diff, info

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    def get_img_ids(self, batch):
        x = batch['file_name']
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        x = self.get_input(batch, self.image_key)
        xrec_aux = None
        if self.use_aux_loss:
            xrec, xrec_aux, qloss, _ = self(x)
        else:
            xrec, qloss, _ = self(x)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train", xrec_aux=xrec_aux)
            self.log("train/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("train/discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec_aux = None
        if self.use_aux_loss:
            xrec, xrec_aux, qloss, _ = self(x)
        else:
            xrec, qloss, _ = self(x)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0, self.global_step,
                                            last_layer=self.get_last_layer(), split="val", xrec_aux=xrec_aux)

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")
        rec_loss = log_dict_ae["val/rec_loss"]
        self.log("val/rec_loss", rec_loss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("val/aeloss", aeloss,
                   prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.ms_quantize.parameters())+
                                  list(self.ms_quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters())+
                                  list(self.upsample.parameters())+
                                  list(self.shared_decoder.parameters())+
                                  list(self.shared_post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))

        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        try:
            img_ids = self.get_img_ids(batch)
            log["file_name"] = img_ids
        except:
            pass
        
        x = x.to(self.device)
        
        if self.use_aux_loss:
            xrec, xrec_aux, _, info = self(x)
            log["reconstructions_aux"] = xrec_aux
        else:
            xrec, _, info = self(x)

        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
            
        log["codebook_info"] = [info[2]]
        log["inputs"] = x
        log["reconstructions"] = xrec

        if len(self.embed_dim) >= 2:
            quant, _, _ = self.encode(x)

            for i in range(len(self.embed_dim)):
                start_channel = sum(self.embed_dim[:i])
                end_channel = sum(self.embed_dim[:i+1])
                    
                print(f'reconstructions channel range [{start_channel}, {end_channel}] (mean/std): ', quant[:, start_channel:end_channel, :, :].mean(), quant[:, start_channel:end_channel, :, :].std())

                noise = torch.zeros((quant.size(0), sum(self.embed_dim), quant.size(2), quant.size(3)), device=self.device)
                samples_down = noise.clone()
                samples_down[:, start_channel:end_channel, :, :] = quant[:, start_channel:end_channel, :, :]
                
                x_samples_down = self.decode(samples_down)
                log[f"reconstructions_{start_channel}_{end_channel}"] = x_samples_down
                
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class VQModelInterface(MSFPNVQModel):
    def __init__(self, embed_dim, channel_range=[], *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim
        self.channel_range = channel_range

    def encode(self, x):
        h_ms = self.encoder(x)

        h_ms = h_ms[::-1]
        prev_h = []
        h_out = []
        for ii in range(len(h_ms)):

            if len(prev_h) != 0:
                for j in range(ii):
                    prev_h[j] = self.upsample[ii-1](prev_h[j])
                    prev_h[j] = self.shared_post_quant_conv[ii-1](prev_h[j])
                
                quant = torch.cat((*prev_h[:ii], h_ms[ii]), dim=1)
                quant = self.shared_decoder[ii-1](quant)
                # quant = quant + prev_h
            else:
                quant = h_ms[ii]

            h = self.ms_quant_conv[ii](quant)
            h_out.append(h)
            quant, emb_loss, info = self.ms_quantize[ii](h)

            prev_h.append(quant)

        if len(self.channel_range) == 2:

            h_out = h_out[self.channel_range[0]//self.embed_dim[0]:self.channel_range[1]//self.embed_dim[0]]

            h_out = h_out[::-1]
            # # upsample each resolutions
            for i in range(len(h_out)):
                for t in range(i):
                    h_out[i] = F.interpolate(h_out[i], scale_factor=2)
            h_out = h_out[::-1]

            h_out = torch.cat(h_out, dim=1) # channel-wise concate

        else:
            h_out = h_out[::-1]
            # # upsample each resolutions
            for i in range(len(h_out)):
                for t in range(i):
                    h_out[i] = F.interpolate(h_out[i], scale_factor=2)
            h_out = h_out[::-1]

            h_out = torch.cat(h_out, dim=1) # channel-wise concate

        return h_out

    def decode(self, h_in, force_not_quantize=False, return_code=False):
        h = h_in.clone()
        
        h_ms = []
        start = 0
        for i in range(len(self.embed_dim)):
            h_ms.append(h[:, start:start+self.embed_dim[i], :, :])
            start += self.embed_dim[i]

        qaunt_ms = []
        code = []
        for ii in range(len(h_ms)):
            quant, emb_loss, info = self.ms_quantize[ii](h_ms[ii])
            qaunt_ms.append(quant)
            code.append(info[2].reshape(len(h_in), -1).tolist())

        qaunt_ms = qaunt_ms[::-1]
        quant = torch.cat(qaunt_ms, dim=1) # channel-wise concate

        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        if return_code:
            return dec, code
        return dec
