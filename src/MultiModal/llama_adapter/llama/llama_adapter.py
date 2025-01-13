# github@OpenGVLab,repo:LLaMA-Adapter, https://github.com/OpenGVLab/LLaMA-Adapter.git

from .generation import sample_top_p
from .model import ModelArgs, Transformer
from .tokenizer import Tokenizer
from ... import clip
import torch
import torch.nn as nn
import os
import json
from pathlib import Path
from timm.models.vision_transformer import Block
import time


class LlamaAdapter(nn.Module):
    def __init__(self, llama_ckpt_dir, llama_tokenizer,
                 max_seq_len=512, max_batch_size=1,
                 clip_model='ViT-L/14',
                 v_embed_dim=768, v_depth=8,
                 v_num_heads=16, v_mlp_ratio=4.0,
                 query_len=10, query_layer=31,
                 device="mps", 
                 phase="finetune"):
        """_summary_

        Args:
            llama_ckpt_dir (str): the llama ckpt dir
            llama_tokenizer (str): llama tokenizer dir
            max_seq_len (int, optional): the sequences length. Defaults to 512.
            max_batch_size (int, optional): _description_. Defaults to 1.
            clip_model (str, optional): _description_. Defaults to 'ViT-L/14'.
            v_embed_dim (int, optional): _description_. Defaults to 768.
            v_depth (int, optional): _description_. Defaults to 8.
            v_num_heads (int, optional): _description_. Defaults to 16.
            v_mlp_ratio (float, optional): _description_. Defaults to 4.0.
            query_len (int, optional): _description_. Defaults to 10.
            query_layer (int, optional): _description_. Defaults to 31.
            phase (str, optional): _description_. Defaults to "finetune".
        """
        super().__init__()
        
        with open(os.path.join(llama_ckpt_dir, "params.json"), "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
        )
        self.clip, _ = clip.load(clip_model)
        self.device = device
        clip_dim = self.clip.visual.proj.shape[1]
        self.clip_proj = nn.Linear(clip_dim, v_embed_dim)
        self.clip_proj_norm = nn.LayerNorm(v_embed_dim)

        self.query_len = query_len
        self.query_layer = query_layer

        self.visual_query = nn.Embedding(query_len, v_embed_dim)
        self.visual_blocks = nn.ModuleList([
            Block(v_embed_dim, v_num_heads, v_mlp_ratio, qkv_bias=True)
            for _ in range(v_depth)])
        self.visual_proj = nn.Linear(v_embed_dim, model_args.dim)
        self.visual_proj_norm = nn.LayerNorm(model_args.dim)

        self.adapter_query = nn.Embedding(
            query_len * query_layer, model_args.dim)

        # 4. tokenizer
        self.tokenizer = Tokenizer(model_path=llama_tokenizer)
        model_args.vocab_size = self.tokenizer.n_words
        torch.set_default_dtype(torch.float16)
        self.llama = Transformer(model_args)
        torch.set_default_dtype(torch.float32)

        start_time = time.time()
        ckpts = sorted(Path(llama_ckpt_dir).glob("*.pth"))
        for ckpt in ckpts:
            ckpt = torch.load(ckpt, map_location='cpu')
            self.llama.load_state_dict(ckpt, strict=False)
        
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        del self.clip.transformer

        # self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        self.phase = phase
        self.get_trainable_params(self.phase)

        for name, param in self.named_parameters():
            if param.requires_grad:
               print(f"Trainable param: {name}, {param.shape}, {param.dtype}")

    def get_trainable_params(self, phase='finetune'):
        for name, para in self.named_parameters():
            para.requires_grad = False

        if phase == 'finetune':
            for name, para in self.named_parameters():
                if name.startswith("llama."):
                    if 'norm' in name or 'bias' in name:
                        para.data = para.data.float()
                        para.requires_grad = True

        elif phase == 'pretrain':
            train_param_name = ['gate', 'clip_proj', 'clip_proj_norm', 'visual_query', 'visual_blocks', 'visual_proj', 'visual_proj_norm', 'adapter_query']
            for name, para in self.named_parameters():
                for train_name in train_param_name:
                    if train_name in name:
                        para.data = para.data.float()
                        para.requires_grad = True

        elif phase == 'inference':
            pass
        
        else:
            raise ValueError(f"Unknown model phase: {phase}")
    
    def clip_encode_image(self, x: torch.Tensor):
        # modified from CLIP
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1,
                      x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # preserve all spatial tokens
        x = self.clip.visual.ln_post(x[:, :, :])

        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj

        return x

    def forward_visual(self, imgs):
        with torch.no_grad():
            clip_feats = self.clip_encode_image(imgs.to(device=self.device))
        
        clip_feats = self.clip_proj_norm(self.clip_proj(clip_feats.float()))

        visual_query = self.visual_query.weight.unsqueeze(0).repeat(len(imgs), 1, 1)
        visual_query = torch.cat([visual_query, clip_feats], dim=1)
        for block in self.visual_blocks:
            visual_query = block(visual_query)

        visual_query = visual_query[:, :self.query_len, :]
        visual_query = self.visual_proj(visual_query)
        visual_query = self.visual_proj_norm(visual_query)

        return visual_query
    
    def forward(self, tokens, imgs, start_pos: int = 0):
        visual_query = self.forward_visual(imgs)

        _bsz, seqlen = tokens.shape

        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[start_pos: start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )

            mask = torch.triu(mask, diagonal=1)
            # a new line append hear, to deal with the bug exist in mps device
            mask = torch.where(torch.isnan(mask), torch.tensor(0.0).to(device=tokens.device), mask)

            
            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

        for layer in self.llama.layers[:-1 * self.query_layer]:
            h = layer(h, 0, freqs_cis, mask)

        adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)
        adapter_index = 0
        for layer in self.llama.layers[-1 * self.query_layer:]:
            dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
            dynamic_adapter = dynamic_adapter + visual_query
            h = layer(h, 0, freqs_cis, mask, dynamic_adapter)
            adapter_index = adapter_index + 1

        h = self.llama.norm(h)
        output = self.llama.output(h)
        output = output[:, :-1, :]

        return output
    
    @torch.inference_mode()
    def forward_inference(self, visual_query, tokens, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.llama.tok_embeddings(tokens)
        freqs_cis = self.llama.freqs_cis.to(h.device)
        freqs_cis = freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )

            mask = torch.triu(mask, diagonal=1)
            mask = torch.where(torch.isnan(mask), torch.tensor(0.0).to(device=tokens.device), mask)
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

        for layer in self.llama.layers[:-1 * self.query_layer]:
            h = layer(h, start_pos, freqs_cis, mask)

        adapter = self.adapter_query.weight.reshape(self.query_layer, self.query_len, -1).unsqueeze(1)
        adapter_index = 0
        for layer in self.llama.layers[-1 * self.query_layer:]:
            dynamic_adapter = adapter[adapter_index].repeat(_bsz, 1, 1)
            dynamic_adapter = dynamic_adapter + visual_query
            h = layer(h, start_pos, freqs_cis, mask, dynamic_adapter)
            adapter_index = adapter_index + 1

        h = self.llama.norm(h)
        output = self.llama.output(h[:, -1, :])

        return output.float()


    @torch.inference_mode()
    def generate(
        self, imgs, prompts,
        max_gen_len: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.75,
    ):
        bsz = len(imgs)
        params = self.llama.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        assert len(imgs) == len(prompts)
        visual_query = self.forward_visual(imgs)

        if isinstance(prompts[0], str):
            prompts = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompts])
        max_prompt_size = max([len(t) for t in prompts])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).long().to(device=imgs.device)

        for k, t in enumerate(prompts):
            tokens[k, : len(t)] = torch.tensor(t).long().to(device=imgs.device)
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            logits = self.forward_inference(visual_query, tokens[:, prev_pos:cur_pos], prev_pos)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)

            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            # trick: early stop if bsz==1
            if bsz == 1 and next_token[0] == self.tokenizer.eos_id:
                break
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):

            # cut to max gen len
            t = t[len(prompts[i]): len(prompts[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(t))

        return decoded