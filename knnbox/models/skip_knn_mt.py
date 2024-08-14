from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np
from torch import Tensor, tensor
import torch
import torch.nn as nn

from fairseq.data import Dictionary
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import (
    TransformerModel,
    TransformerEncoder,
    TransformerDecoder,
)
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq import utils
from knnbox.common_utils import (
    global_vars,
    select_keys_with_pad_mask,
    disable_model_grad,
    enable_module_grad,
    archs,
    repetition_mask,
)
from knnbox.datastore import Datastore, PckDatastore
from knnbox.retriever import Retriever, LinesRetriever
from knnbox.combiner import SkipCombiner, AdaptiveCombiner


@register_model("skip_knn_mt")
class SkipKNNMT(TransformerModel):
    r"""
    more skip knn-mt model.
    """

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)

        # when train metak network, we should disable other module's gradient
        # and only enable the combiner(metak network)'s gradient 
        if args.knn_mode == "train_metak":
            disable_model_grad(self)
            enable_module_grad(self, "combiner")

        elif args.knn_mode == "build_datastore":
            if "datastore" not in global_vars():
                # regist the datastore as a global variable if not exist,
                # because we need access the same datastore in another
                # python file (when traverse the dataset and `add value`)
                global_vars()["datastore"] = Datastore(args.knn_datastore_path)
            self.datastore = global_vars()["datastore"]

    @staticmethod
    def add_args(parser):
        r"""
        add knn-mt related args here
        """
        TransformerModel.add_args(parser)
        parser.add_argument("--knn-mode", choices=["build_datastore", "train_metak", "inference"],
                            help="choose the action mode")
        parser.add_argument("--knn-datastore-path", type=str, metavar="STR",
                            help="the directory of save or load datastore")
        parser.add_argument("--knn-max-k", type=int, metavar="N", default=8,
                            help="The hyper-parameter max k of skip knn-mt")

        parser.add_argument("--knn-combiner-path", type=str, metavar="STR", default="/home/",
                            help="The directory to save/load SkipCombiner")
        parser.add_argument("--build-faiss-index-with-cpu", action="store_true", default=False,
                            help="use faiss-cpu instead of faiss-gpu (useful when gpu memory is small)")

        parser.add_argument("--skip-inference", default=False, action='store_true')

        parser.add_argument("--knn-lambda", type=float, metavar="D", default=0.7,
                            help="The hyper-parameter lambda of vanilla knn-mt")
        parser.add_argument("--knn-temperature", type=float, metavar="D", default=10,
                            help="The hyper-parameter temperature of vanilla knn-mt")
        parser.add_argument("--alphamin", type=float, metavar="D", default=0.4,
                            help="The hyper-parameter alpha min")
        parser.add_argument("--datasetlength", type=float, metavar="D",
                            help="The hyper-parameter dataset length")
        parser.add_argument("--base_model_path", type=str)
        parser.add_argument("--dataset", type=str)
        parser.add_argument('--with-pck', default=False, action='store_true')
        parser.add_argument("--pck-combiner-path", type=str, metavar="STR", default="/home/",
                            help="The directory to save/load pckCombiner")

    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            return_all_hiddens: bool = True,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        if self.datastore is not None:
            non_mask = ~encoder_out.encoder_padding_mask.transpose(0, 1).unsqueeze(dim=-1).expand_as(encoder_out.encoder_out)
            src_key = torch.sum(encoder_out.encoder_out * non_mask, dim=0) / torch.sum(non_mask, dim=0)
            self.datastore["src_keys"].add(src_key)
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens
        )
        return decoder_out

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return SkipKNNMTEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        r"""
        we override this function, replace the TransformerDecoder with SkipKNNMTDecoder
        """
        return SkipKNNMTDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
            position=None,
            **kwargs
    ):
        return self.decoder.get_normalized_probs(
            net_output,
            log_probs,
            sample,
            position,
            **kwargs
        )


class SkipKNNMTEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

    def forward(
            self,
            src_tokens,
            src_lengths,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings
        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,
            src_lengths=None,
        )


class SkipKNNMTDecoder(TransformerDecoder):
    r"""
    Skip knn-mt Decoder, equipped with Datastore, Retriever and SkipCombiner.
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        r"""
        we override this function to create knn-related module.
        In other words, create datastore, retriever and combiner.
        """
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)

        self.update_num = 0
        self.skip_inference = args.skip_inference
        self.skip_retrieve = self.skip_inference

        if args.knn_mode == "inference":
            if args.with_pck:
                self.datastore = PckDatastore.load(args.knn_datastore_path, load_list=["vals"], load_network=True)
            else:
                self.datastore = Datastore.load(args.knn_datastore_path, load_list=["keys", "vals"])
            self.datastore.load_faiss_index("keys")
            self.retriever = Retriever(datastore=self.datastore, k=args.knn_max_k)
            self.combiner = SkipCombiner.load(args.knn_combiner_path)
            min_threshold = args.alphamin
            self.skip_thresholds = (min_threshold + (0.5-min_threshold) * np.power((np.clip((np.arange(0, 1024)-1) /
                                    args.datasetlength, 0, 1)), 2))
        elif args.knn_mode == "build_datastore":
            if "datastore" not in global_vars():
                # regist the datastore as a global variable if not exist,
                # because we need access the same datastore in another
                # python file (when traverse the dataset and `add value`)
                global_vars()["datastore"] = Datastore(args.knn_datastore_path)
            self.datastore = global_vars()["datastore"]

            vocab = Dictionary.load(f'{args.base_model_path}/fairseq-vocab.txt')
            output_projection_ckp = torch.load(f'{args.base_model_path}/output_projection.pt')
            self.output_projection = nn.Linear(1024, len(vocab), bias=False)
            self.output_projection.load_state_dict(output_projection_ckp)
        else:
            raise NotImplementedError

        self.return_list = ["vals", "distances"]

        # In order to count skip
        self.retrieve_count = []
        self.deno_count = []

    def forward(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            features_only: bool = False,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            src_lengths: Optional[Any] = None,
            return_all_hiddens: bool = False,
    ):
        r"""
        we overwrite this function to do something else besides forward the TransformerDecoder.
        
        when the action mode is `building datastore`, we save keys to datastore.
        when the action mode is `inference`, we retrieve the datastore with hidden state.
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )
        if self.args.knn_mode == "build_datastore":
            keys = select_keys_with_pad_mask(x, self.datastore.get_pad_mask())
            # save half precision keys
            self.datastore["keys"].add(keys.half())
            nmt_prob = torch.softmax(self.output_projection(keys), dim=-1)
            _, nmt_max_token = torch.max(nmt_prob, dim=-1, keepdim=True)

            attns_max_prob, attns_max_token = torch.max(extra['attn'][0], dim=2)
            attns_max_prob = select_keys_with_pad_mask(attns_max_prob.unsqueeze(dim=-1), self.datastore.get_pad_mask())

            self.datastore["attns"].add(attns_max_prob)
        extra.update({"last_hidden": x})
        extra.update({"prev_output_tokens": prev_output_tokens})

        if not features_only:
            x = self.output_layer(x)
        return x, extra

    """
        A scriptable subclass of this class has an extract_features method and calls
        super().extract_features, but super() is not supported in torchscript. Aa copy of
        this function is made to be used in the subclass instead.
        """

    def extract_features_scriptable(
            self,
            prev_output_tokens,
            encoder_out: Optional[EncoderOut] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
            specific_layer: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None

        specific_layer_features: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=True,
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

            if specific_layer is not None and specific_layer == (idx + 1):
                specific_layer_features = x.transpose(0, 1)


        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states, "specific_layer_features": specific_layer_features,
                   "encoder_out": encoder_out}



    # @profile
    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
            position=None,
            **kwargs
    ):
        r"""
        we overwrite this function to change the probability calculation process.
        step 1. 
            calculate the knn probability based on retrieved resultes
        step 2.
            combine the knn probability with NMT's probability 
        """
        if self.args.knn_mode == "inference":
            # scores = kwargs['scores']
            query = net_output[1]["last_hidden"].squeeze(1)
            nmt_prob = utils.softmax(net_output[0].squeeze(1), dim=-1, onnx_trace=self.onnx_trace)

            if self.skip_retrieve:
                prev_output_tokens = net_output[1]['prev_output_tokens']
                attns_max_prob, attns_max_token = torch.max(net_output[1]['attn'][0], dim=2)

                norm = torch.norm(net_output[1]["last_hidden"].squeeze(1), dim=1)

                knn_alpha = self.combiner.meta_network(nmt_prob, attn=attns_max_prob, norm=norm).squeeze()
                # move to cpu, and use numpy.nonzero to get the retrieve_index
                knn_alpha_np = knn_alpha.cpu().numpy()

                skip_threshold = self.skip_thresholds[prev_output_tokens.shape[1]]
                retrieve_index_np = (knn_alpha_np > skip_threshold).nonzero()[0]

                retrieve_index = torch.tensor(retrieve_index_np, device=knn_alpha.device)

                retrieve_num = retrieve_index.numel()
                deno_num = knn_alpha.numel()
                self.retrieve_count.append(retrieve_num)
                self.deno_count.append(deno_num)
                if retrieve_num == 0:
                    return nmt_prob.log().unsqueeze(1)

                retrieve_query = query.index_select(dim=0, index=retrieve_index)
                if self.args.with_pck:
                    self.retriever.retrieve(self.datastore.vector_reduct(retrieve_query), return_list=self.return_list)
                else:
                    self.retriever.retrieve(
                        retrieve_query,
                        return_list=self.return_list,
                        bsz=query.size(0),
                        retrieve_index=retrieve_index,
                    )

            else:
                self.retriever.retrieve(query, return_list=self.return_list, bsz=query.size(0))

                knn_alpha = None
                retrieve_index = None
            combined_prob = self.combiner.forward(
                nmt_prob=nmt_prob,
                knn_dist=self.retriever.results["distances"],
                knn_tgt=self.retriever.results["vals"],
                knn_alpha=knn_alpha,
                retrieve_index=retrieve_index,
            )
            return combined_prob.log().unsqueeze(1)

        else:
            return super().get_normalized_probs(net_output, log_probs, sample)

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self.update_num = num_updates


r""" Define some me knn-mt's arch.
     arch name format is: knn_mt_type@base_model_arch
"""


@register_model_architecture("skip_knn_mt", "skip_knn_mt@transformer")
def base_architecture(args):
    archs.base_architecture(args)


@register_model_architecture("skip_knn_mt", "skip_knn_mt@transformer_iwslt_de_en")
def transformer_iwslt_de_en(args):
    archs.transformer_iwslt_de_en(args)


@register_model_architecture("skip_knn_mt", "skip_knn_mt@transformer_wmt_en_de")
def transformer_wmt_en_de(args):
    archs.base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture("skip_knn_mt", "skip_knn_mt@transformer_vaswani_wmt_en_de_big")
def transformer_vaswani_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture("skip_knn_mt", "skip_knn_mt@transformer_vaswani_wmt_en_fr_big")
def transformer_vaswani_wmt_en_fr_big(args):
    archs.transformer_vaswani_wmt_en_fr_big(args)


@register_model_architecture("skip_knn_mt", "skip_knn_mt@transformer_wmt_en_de_big")
def transformer_wmt_en_de_big(args):
    archs.transformer_vaswani_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture("skip_knn_mt", "skip_knn_mt@transformer_wmt_en_de_big_t2t")
def transformer_wmt_en_de_big_t2t(args):
    archs.transformer_wmt_en_de_big_t2t(args)


@register_model_architecture("skip_knn_mt", "skip_knn_mt@transformer_wmt19_de_en")
def transformer_wmt19_de_en(args):
    archs.transformer_wmt19_de_en(args)


@register_model_architecture("skip_knn_mt", "skip_knn_mt@transformer_zh_en")
def transformer_my_zh_en(args):
    archs.transformer_my_zh_en(args)
