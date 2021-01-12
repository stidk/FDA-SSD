# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import math
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

from ppdet.experimental import mixed_precision_global_state
from ppdet.core.workspace import register
from ppdet.modeling.ops import SSDOutputDecoder

__all__ = ['SSD']


@register
class SSD(object):
    """
    Single Shot MultiBox Detector, see https://arxiv.org/abs/1512.02325

    Args:
        backbone (object): backbone instance
        multi_box_head (object): `MultiBoxHead` instance
        output_decoder (object): `SSDOutputDecoder` instance
        num_classes (int): number of output classes
    """

    __category__ = 'architecture'
    __inject__ = ['backbone', 'multi_box_head', 'output_decoder', 'fpn']
    __shared__ = ['num_classes']

    def __init__(self,
                 backbone,
                 fpn='FPN',
                 multi_box_head='MultiBoxHead',
                 output_decoder=SSDOutputDecoder().__dict__,
                 num_classes=21):
        super(SSD, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.multi_box_head = multi_box_head
        self.num_classes = num_classes
        self.output_decoder = output_decoder
        if isinstance(output_decoder, dict):
            self.output_decoder = SSDOutputDecoder(**output_decoder)

    def build(self, feed_vars, mode='train'):
        im = feed_vars['image']
        if mode == 'train' or mode == 'eval':
            gt_bbox = feed_vars['gt_bbox']
            gt_class = feed_vars['gt_class']

        mixed_precision_enabled = mixed_precision_global_state() is not None
        # cast inputs to FP16
        if mixed_precision_enabled:
            im = fluid.layers.cast(im, 'float16')

        # backbone
        body_feats = self.backbone(im)
        body_feat_names1 = list(body_feats.keys())
        body_feats1 = [body_feats[name] for name in body_feat_names1]
        # print(body_feats1)
        # body_feats1 = []
        body_feats_w = []
        body_feats2 = []

        # body_feats2 = []
        # body_feats3 = []
        num_of_layers = [256, 512, 1024, 2048]
        i = 0
        for body_feat in body_feats1:
            # print(type(body_feat))
            name1 = 'se' + str(i)
            # name2 = 'se_2' + str(i)
            body_feats_w += [self._squeeze_excitation(body_feat,num_of_layers[i],name = name1)]
            i += 1
        body_feast_all_w = fluid.layers.concat([body_feats_w[0], body_feats_w[1], body_feats_w[2], body_feats_w[3]], axis=1)
        # print(len(num_of_layers))
        for i in range(len(num_of_layers)):
            body_feats_w2 = fluid.layers.fc(
            input=body_feast_all_w,
            size=num_of_layers[i],
            act='sigmoid')
            body_feats2 += [fluid.layers.elementwise_mul(x=body_feats1[i], y=body_feats_w2, axis=0)]
        # print(body_feats2)
            # body_feats2 += [self.spatialgate(body_feat,num_of_layers[i],name = name2)]
            # body_feats1 += [fluid.layers.elementwise_add(x = body_feat, y=s, act='relu', name=name1 + ".add.output.5")]
            # i += 1
        # for i in range(len(body_feats1)):
        #     body_feats3 += [body_feats1[i] + body_feats2[i]]
        # if isinstance(body_feats, OrderedDict):
        #     body_feat_names = list(body_feats.keys())
        #     body_feats = [body_feats[name] for name in body_feat_names]
        #     # print(1)
        #     new_feats = []
        #     new_feats = body_feats
        #     # print(body_feats)
        #     new_feats_change = self.BasicConv(body_feats[0], 256, filter_size = 1, down_size = 38)
        #     new_feats_change2 = self.BasicConv(body_feats[1], 256, filter_size = 1)
        #     # print(new_feats_change, new_feats_change2)
        #     new_feats[0] = fluid.layers.concat([new_feats_change, new_feats_change2], axis = 1)
        #     # print(3)
        #     del(new_feats[1])
        #     print(new_feats)
        body_feats2 = OrderedDict([('new_feats{}'.format(idx),feat)
                                for idx, feat in enumerate(body_feats2)])
        if self.fpn is not None:
            body_feats3, spatial_scale = self.fpn.get_output(body_feats2)

        if isinstance(body_feats3, OrderedDict):
            body_feat_names = list(reversed(list(body_feats3.keys())))
            body_feats = [body_feats3[name] for name in body_feat_names]
            new_feats = []
            new_feats = body_feats
            # print(body_feats)
            new_feats_change = self.BasicConv(body_feats[0], 256, filter_size = 1, down_size = 38)
            new_feats_change2 = self.BasicConv(body_feats[1], 256, filter_size = 1)
            # print(new_feats_change, new_feats_change2)
            new_feats[0] = fluid.layers.concat([new_feats_change, new_feats_change2], axis = 1)
            # print(3)
            del(new_feats[1])
            body_feats = new_feats
        #     # new_feat_two = []
        #     # new_feat_two = body_feats
        #     # new_feats_change3 = self.BasicConv(body_feats[-2], 256, filter_size = 1, down_size = 1)
        #     # new_feats_change4 = self.BasicConv(body_feats[-1], 256, filter_size = 1)
        #     # new_feat_two[-1] = fluid.layers.concat([new_feats_change3, new_feats_change4], axis = 1)
        #     # del(new_feat_two[-2])
        #     # body_feats = new_feat_two
        # body_feat_names = list(reversed(list(body_feats.keys())))
        # body_feats = [body_feats[name] for name in body_feat_names]
        # del(body_feats[0])
        del(body_feats[-2])
        # body_feats.append(fluid.layers.conv2d(body_feats[-1], 256, 3, 1, 0,act = 'swish')) 
        print(body_feats)
        # cast features back to FP32
        # body_feats1 = []
        # body_feats2 = []
        # body_feats3 = []
        # num_of_layers = [512,256,256,256,256,256]
        # i = 0
        # for body_feat in body_feats:
        #     # print(type(body_feat))
        #     name1 = 'se' + str(i)
        #     name2 = 'se_2' + str(i)
        #     body_feats1 += [self._squeeze_excitation(body_feat,num_of_layers[i],name = name1)]
        #     body_feats2 += [self.spatialgate(body_feat,num_of_layers[i],name = name2)]
        #     # body_feats1 += [fluid.layers.elementwise_add(x = body_feat, y=s, act='relu', name=name1 + ".add.output.5")]
        #     i += 1
        # for i in range(len(body_feats1)):
        #     body_feats3 += [body_feats1[i] + body_feats2[i]]
        if mixed_precision_enabled:
            body_feats = [fluid.layers.cast(v, 'float32') for v in body_feats]

        locs, confs, box, box_var = self.multi_box_head(
            inputs=body_feats, image=im, num_classes=self.num_classes)

        if mode == 'train':
            loss = fluid.layers.ssd_loss(locs, confs, gt_bbox, gt_class, box,
                                         box_var)
            loss = fluid.layers.reduce_sum(loss)
            return {'loss': loss}
        else:
            pred = self.output_decoder(locs, confs, box, box_var)
            return {'bbox': pred}
    def spatialgate(self, input, num_channels, name =None):
        mixed_precision_enabled = mixed_precision_global_state() is not None
        pool1 = fluid.layers.pool2d(
            input=input,
            pool_size=0,
            pool_type='max',
            global_pooling=True,
            use_cudnn=mixed_precision_enabled)
        pool2 = fluid.layers.pool2d(
            input=input,
            pool_size=0,
            pool_type='avg',
            global_pooling=True,
            use_cudnn=mixed_precision_enabled)
        pool_all = fluid.layers.concat([pool1,pool2],axis = 1)
        conv_1 = self.BasicConv(pool_all, num_channels, filter_size = 7, padding = 3,act = None)
        scale = fluid.layers.sigmoid(conv_1)
        scale = fluid.layers.elementwise_mul(x=input, y=scale, axis=0)
        return scale
    def _squeeze_excitation(self, input, num_channels, name=None):
        mixed_precision_enabled = mixed_precision_global_state() is not None
        pool = fluid.layers.pool2d(
            input=input,
            pool_size=0,
            pool_type='avg',
            global_pooling=True,
            use_cudnn=mixed_precision_enabled)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        squeeze = fluid.layers.fc(
            input=pool,
            size=int(num_channels / 16),
            act='relu',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_sqz_weights'),
            bias_attr=ParamAttr(name=name + '_sqz_offset'))
        stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
        excitation = fluid.layers.fc(
            input=squeeze,
            size=num_channels,
            act='sigmoid',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_exc_weights'),
            bias_attr=ParamAttr(name=name + '_exc_offset'))
        scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
        return excitation
    def _squeeze_excitation1(self, input, num_channels, name=None):
        mixed_precision_enabled = mixed_precision_global_state() is not None
        pool = fluid.layers.pool2d(
            input=input,
            pool_size=0,
            pool_type='max',
            global_pooling=True,
            use_cudnn=mixed_precision_enabled)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        squeeze = fluid.layers.fc(
            input=pool,
            size=int(num_channels / 16),
            act='relu',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_sqz_weights'),
            bias_attr=ParamAttr(name=name + '_sqz_offset'))
        stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
        excitation = fluid.layers.fc(
            input=squeeze,
            size=num_channels,
            act='sigmoid',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_exc_weights'),
            bias_attr=ParamAttr(name=name + '_exc_offset'))
        scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
        return scale
    def _inputs_def(self, image_shape):
        im_shape = [None] + image_shape
        # yapf: disable
        inputs_def = {
            'image':        {'shape': im_shape,  'dtype': 'float32', 'lod_level': 0},
            'im_id':        {'shape': [None, 1], 'dtype': 'int64',   'lod_level': 0},
            'gt_bbox':      {'shape': [None, 4], 'dtype': 'float32', 'lod_level': 1},
            'gt_class':     {'shape': [None, 1], 'dtype': 'int32',   'lod_level': 1},
            'im_shape':     {'shape': [None, 3], 'dtype': 'int32',   'lod_level': 0},
            'is_difficult': {'shape': [None, 1], 'dtype': 'int32',   'lod_level': 1},
        }
        # yapf: enable
        return inputs_def
    def BasicConv(self, input, num_filters, filter_size, stride = 1, padding = 0, dilation = 1, act = 'swish', bn = True, down_size = 0, groups = 1):
        conv_1 = self._conv_layer(input = input, num_filters = num_filters, filter_size = filter_size, stride = stride, padding = padding, dilation = dilation, act = None)
        if bn:
            conv_1 = fluid.layers.batch_norm(input = conv_1, momentum = 0.01)
            # print(1)
        if down_size > 0:
            conv_1 = fluid.layers.interpolate(conv_1, out_shape =(down_size, down_size), resample='BILINEAR', align_corners=True)
        if act =='swish':
            conv_1 = fluid.layers.swish(conv_1)
            # print(2)
        return conv_1
    def _conv_layer(self,
                    input,
                    num_filters,
                    filter_size,
                    stride = 1,
                    padding = 0,
                    dilation = 1,
                    act='relu',
                    groups = 1
                    ):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            act=act,
            )
        
        return conv
    def build_inputs(
            self,
            image_shape=[3, None, None],
            fields=['image', 'im_id', 'gt_bbox', 'gt_class'],  # for train
            use_dataloader=True,
            iterable=False):
        inputs_def = self._inputs_def(image_shape)
        feed_vars = OrderedDict([(key, fluid.data(
            name=key,
            shape=inputs_def[key]['shape'],
            dtype=inputs_def[key]['dtype'],
            lod_level=inputs_def[key]['lod_level'])) for key in fields])
        loader = fluid.io.DataLoader.from_generator(
            feed_list=list(feed_vars.values()),
            capacity=16,
            use_double_buffer=True,
            iterable=iterable) if use_dataloader else None
        return feed_vars, loader

    def train(self, feed_vars):
        return self.build(feed_vars, 'train')

    def eval(self, feed_vars):
        return self.build(feed_vars, 'eval')

    def test(self, feed_vars, exclude_nms=False):
        assert not exclude_nms, "exclude_nms for {} is not support currently".format(
            self.__class__.__name__)
        return self.build(feed_vars, 'test')

    def is_bbox_normalized(self):
        # SSD use output_decoder in output layers, bbox is normalized
        # to range [0, 1], is_bbox_normalized is used in eval.py and infer.py
        return True
