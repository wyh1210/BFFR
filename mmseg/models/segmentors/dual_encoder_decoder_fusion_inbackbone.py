# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .dual_base import DualBaseSegmentor


@SEGMENTORS.register_module()
class DualEncoderDecoderFusionInbackbone(DualBaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 frozen=None,
                 neck=None,
                 fusion=None,
                 nopretrain=None,
                #  auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(DualEncoderDecoderFusionInbackbone, self).__init__(init_cfg)
        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
        backbone_rgb = backbone
        backbone_ir = backbone
        if nopretrain=='rgb':
            backbone_rgb.pretrained = None
        elif nopretrain=='ir':
            backbone_ir.pretrained = None
        self.backbone_rgb = builder.build_backbone(backbone_rgb)
        self.backbone_ir = builder.build_backbone(backbone_ir)
        if frozen is not None:
            assert frozen in ['rgb', 'ir', 'all'], 'frozen must be in one of [rgb, ir, all].'
            if frozen=='rgb':
                self.backbone_rgb.frozen_all()
            elif frozen=='ir':
                self.backbone_ir.frozen_all()
            elif frozen=='all':
                self.backbone_ir.frozen_all()
                self.backbone_rgb.frozen_all()
        # self.weigtsir_former = self.backbone_ir.state_dict()
        # self.weigtsrgb_former = self.backbone_rgb.state_dict()
        self.fusion_module = builder.build_fusion(fusion)

        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        # self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    # def _init_auxiliary_head(self, auxiliary_head):
    #     """Initialize ``auxiliary_head``"""
    #     if auxiliary_head is not None:
    #         if isinstance(auxiliary_head, list):
    #             self.auxiliary_head = nn.ModuleList()
    #             for head_cfg in auxiliary_head:
    #                 self.auxiliary_head.append(builder.build_head(head_cfg))
    #         else:
    #             self.auxiliary_head = builder.build_head(auxiliary_head)

    def extract_feat(self, img_rgb, img_ir):
        """Extract features from images."""
        # print('extract features.')
        # print(self.backbone_ir.stem.parameters()[1])
        # flag = 'Equal'
        # print(self.backbone_ir.state_dict()['stem.0.weight'].device)
        # print(self.weigtsir_former['stem.0.weight'].device)

        # for i in self.backbone_ir.state_dict():
        #     if True in (self.backbone_ir.state_dict()[i] != self.weigtsir_former[i].cuda(2)):
        #         # print(self.backbone_ir.state_dict()[i].device)
        #         print('backbone ir is updated.')
        #         break
        # for i in self.backbone_ir.state_dict():
        #     if True in (self.backbone_rgb.state_dict()[i] != self.weigtsrgb_former[i].cuda(2)):
        #         print('backbone rgb is updated.')
        #         break
        # self.weigtsir_former = self.backbone_ir.state_dict()
        # self.weigtsrgb_former = self.backbone_rgb.state_dict()

        # if self.backbone_rgb.state_dict() != self.weigtsrgb_former:
        #     print('backbonergb is updated.')
        # for param_tensor in self.backbone_ir.state_dict():
        #     weights1 = self.backbone_ir.state_dict()[param_tensor]
        #     weights2 = self.backbone_rgb.state_dict()[param_tensor]
        #     if False in (weights1 == weights2):
        #         flag = 'Vary'
            # print(param_tensor)
        # print(flag)

        # print(self.backbone_ir.frozen_stages)
        # print(self.backbone_rgb.frozen_stages)
        # weights1 = self.backbone_ir.state_dict()['stem.0.weight']
        # weights2 = self.backbone_rgb.state_dict()['stem.0.weight']
        # print(False in (weights1==weights2))


        # exit()
            # print(self.backbone_ir.state_dict()['layer4.2.bn3.weight'])
        # for p in self.backbone_ir.stem.parameters():
        #     print(p)
        # for p in self.backbone_ir.stem.parameters():
        #     print(p)
        f_rgb = img_rgb
        f_ir = img_ir
        outs = []
        for i in range(len(self.backbone_ir.layers)):
            # print(self.backbone_rgb.forward_layer())
            f_rgb, hw_shape= self.backbone_rgb.forward_layer(f_rgb, i)
            f_ir, _ = self.backbone_ir.forward_layer(f_ir, i)
            # print(self.fusion_module)
            # print(f_rgb, f_ir, i)
            f_rgb, f_ir, f_out = self.fusion_module.forward_layer(f_rgb, f_ir, i, hw_shape)
            outs.append(f_out)
        # for o in outs:
        #     print(f'o.shape:{o.shape}')
        # exit()
        return outs

    def encode_decode(self, img_rgb, img_ir, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img_rgb, img_ir)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img_rgb.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    # def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
    #     """Run forward function and calculate loss for auxiliary head in
    #     training."""
    #     losses = dict()
    #     if isinstance(self.auxiliary_head, nn.ModuleList):
    #         for idx, aux_head in enumerate(self.auxiliary_head):
    #             loss_aux = aux_head.forward_train(x, img_metas,
    #                                               gt_semantic_seg,
    #                                               self.train_cfg)
    #             losses.update(add_prefix(loss_aux, f'aux_{idx}'))
    #     else:
    #         loss_aux = self.auxiliary_head.forward_train(
    #             x, img_metas, gt_semantic_seg, self.train_cfg)
    #         losses.update(add_prefix(loss_aux, 'aux'))

    #     return losses

    def forward_dummy(self, img_rgb, img_ir):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img_rgb, img_ir, None)

        return seg_logit

    def forward_train(self, img_rgb, img_ir, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img_rgb, img_ir)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        # if self.with_auxiliary_head:
        #     loss_aux = self._auxiliary_head_forward_train(
        #         x, img_metas, gt_semantic_seg)
        #     losses.update(loss_aux)
        # print(losses)
        # exit()
        return losses
        
    # TODO refactor
    def slide_inference(self, img_rgb, img_ir, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        # print(img_meta)
        # exit()
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img_rgb.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img_rgb.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img_rgb.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img_rgb = img_rgb[:, :, y1:y2, x1:x2]
                crop_img_ir = img_ir[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img_rgb, crop_img_ir, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img_rgb.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img_rgb, img_ir, img_meta, rescale):
        """Inference with full image."""

        seg_logit = self.encode_decode(img_rgb, img_ir, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img_rgb.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img_rgb, img_ir, img_meta, rescale):
        # print(img_rgb.shape)
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """

        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img_rgb, img_ir, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img_rgb, img_ir, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))
        # print(output)
        # print(output.shape)
        # exit()
        return output

    def simple_test(self, img_rgb, img_ir, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img_rgb, img_ir, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs1, imgs2, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs1[0], imgs2[0], img_metas[0], rescale)
        for i in range(1, len(imgs1)):
            cur_seg_logit = self.inference(imgs1[i], imgs2[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs1)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred