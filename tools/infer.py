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

import os, sys
# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

import glob
import numpy as np
import six
from PIL import Image
import cv2

from paddle import fluid

from ppdet.core.workspace import load_config, merge_config, create

from ppdet.utils.eval_utils import parse_fetches
from ppdet.utils.cli import ArgsParser
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.visualizer import visualize_results
import ppdet.utils.checkpoint as checkpoint

from ppdet.data.reader import create_reader

import logging
FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def get_save_image_name(output_dir, image_path):
    """
    Get save image name from source image path.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image_name = os.path.split(image_path)[-1]
    name, ext = os.path.splitext(image_name)
    return os.path.join(output_dir, "{}".format(name)) + ext


def get_test_images(infer_dir, infer_img):
    """
    Get image path list in TEST mode
    """
    assert infer_img is not None or infer_dir is not None, \
        "--infer_img or --infer_dir should be set"
    assert infer_img is None or os.path.isfile(infer_img), \
            "{} is not a file".format(infer_img)
    assert infer_dir is None or os.path.isdir(infer_dir), \
            "{} is not a directory".format(infer_dir)

    # infer_img has a higher priority
    if infer_img and os.path.isfile(infer_img):
        return [infer_img]

    images = set()
    infer_dir = os.path.abspath(infer_dir)
    assert os.path.isdir(infer_dir), \
        "infer_dir {} is not a directory".format(infer_dir)
    exts = ['jpg', 'jpeg', 'png', 'bmp']
    exts += [ext.upper() for ext in exts]
    for ext in exts:
        images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
    images = list(images)

    assert len(images) > 0, "no image found in {}".format(infer_dir)
    logger.info("Found {} inference images in total.".format(len(images)))

    return images


def draw(results, save_path = 'new_featrues', img_num = 100):
    """
    可视化对应层结果
    :param save_path: 图像保存时的文件路径以及文件名
    :param results: 经卷积/池化层后的运算结果，格式为NCHW - 例： 128(batch size), 32(num_filters=16), 30(H), 15(W)
    :param img_num: 采样数
    """

    def fix_value(ipt):
        pix_max = np.max(ipt)
        pix_min = np.min(ipt)
        # base_value = np.abs(pix_min) + np.abs(pix_max)
        # base_rate = 255 / base_value
        # pix_left = base_rate * pix_min
        # ipt = ipt * base_rate - pix_left
        # ipt[ipt < 0] = 0.
        # ipt[ipt > 255] = 1.
        ipt = ((ipt - pix_min) / (pix_max - pix_min + 0.000001))*255
        return ipt

    im_list = None
    for result_i, result in enumerate(results):
        if result_i == img_num - 1:
            break
        result = np.sum(result, axis=0)
        # result = np.sum(result, axis=0)
        print(result.shape)
        im = fix_value(result)
        if im_list is None:
            im_list = im
        else:
            im_list = np.append(im_list, im, axis=1)
    im=im.astype(np.uint8)
    heat_img = cv2.applyColorMap(im, 4)
    im1 = Image.fromarray(np.array(heat_img).astype('uint8')).convert("RGB")
    im = Image.fromarray(np.array(im_list).astype('uint8')).convert("RGB")
    # im = im.resize((256, 256))
    im1.save(save_path + ".jpg")
    return np.array(im1)
def main():
    cfg = load_config(FLAGS.config)

    merge_config(FLAGS.opt)
    check_config(cfg)
    # check if set use_gpu=True in paddlepaddle cpu version
    check_gpu(cfg.use_gpu)
    # check if paddlepaddle version is satisfied
    check_version()

    main_arch = cfg.architecture

    dataset = cfg.TestReader['dataset']

    test_images = get_test_images(FLAGS.infer_dir, FLAGS.infer_img)
    dataset.set_images(test_images)

    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    model = create(main_arch)

    startup_prog = fluid.Program()
    infer_prog = fluid.Program()
    with fluid.program_guard(infer_prog, startup_prog):
        with fluid.unique_name.guard():
            inputs_def = cfg['TestReader']['inputs_def']
            inputs_def['iterable'] = True
            feed_vars, loader = model.build_inputs(**inputs_def)
            test_fetches = model.test(feed_vars)
    infer_prog = infer_prog.clone(True)

    reader = create_reader(cfg.TestReader, devices_num=1)
    loader.set_sample_list_generator(reader, place)

    exe.run(startup_prog)
    if cfg.weights:
        checkpoint.load_params(exe, infer_prog, cfg.weights)

    # parse infer fetches
    assert cfg.metric in ['COCO', 'VOC', 'OID', 'WIDERFACE'], \
            "unknown metric type {}".format(cfg.metric)
    extra_keys = []
    if cfg['metric'] in ['COCO', 'OID']:
        extra_keys = ['im_info', 'im_id', 'im_shape']
    if cfg['metric'] == 'VOC' or cfg['metric'] == 'WIDERFACE':
        extra_keys = ['im_id', 'im_shape']
    keys, values, _ = parse_fetches(test_fetches, infer_prog, extra_keys)
    values.append('concat_0.tmp_0')
    values.append('tmp_3')
    values.append('res2c.add.output.5.tmp_1')
    values.append('res3d.add.output.5.tmp_1')
    values.append('fpn_7.tmp_2')
    keys.append('no_se')
    keys.append('is_se')
    keys.append('after_backbone')
    keys.append('no_se_1')
    keys.append('is_se_1')
    print(values,keys)

    # parse dataset category
    if cfg.metric == 'COCO':
        from ppdet.utils.coco_eval import bbox2out, mask2out, get_category_info
    if cfg.metric == 'OID':
        from ppdet.utils.oid_eval import bbox2out, get_category_info
    if cfg.metric == "VOC":
        from ppdet.utils.voc_eval import bbox2out, get_category_info
    if cfg.metric == "WIDERFACE":
        from ppdet.utils.widerface_eval_utils import bbox2out, lmk2out, get_category_info

    anno_file = dataset.get_anno()
    with_background = dataset.with_background
    use_default_label = dataset.use_default_label

    clsid2catid, catid2name = get_category_info(anno_file, with_background,
                                                use_default_label)

    # whether output bbox is normalized in model output layer
    is_bbox_normalized = False
    if hasattr(model, 'is_bbox_normalized') and \
            callable(model.is_bbox_normalized):
        is_bbox_normalized = model.is_bbox_normalized()

    # use VisualDL to log image
    if FLAGS.use_vdl:
        assert six.PY3, "VisualDL requires Python >= 3.5"
        from visualdl import LogWriter
        vdl_writer = LogWriter(FLAGS.vdl_log_dir)
        vdl_image_step = 0
        vdl_image_frame = 0  # each frame can display ten pictures at most.

    imid2path = dataset.get_imid2path()
    for iter_id, data in enumerate(loader()):
        outs = exe.run(infer_prog,
                       feed=data,
                       fetch_list=values,
                       return_numpy=False)
        res = {
            k: (np.array(v), v.recursive_sequence_lengths())
            for k, v in zip(keys, outs)
        }
        print(np.array(outs[-1]).shape)
        logger.info('Infer iter {}'.format(iter_id))
        if 'TTFNet' in cfg.architecture:
            res['bbox'][1].append([len(res['bbox'][0])])
        if 'CornerNet' in cfg.architecture:
            from ppdet.utils.post_process import corner_post_process
            post_config = getattr(cfg, 'PostProcess', None)
            corner_post_process(res, post_config, cfg.num_classes)

        bbox_results = None
        mask_results = None
        lmk_results = None
        if 'bbox' in res:
            bbox_results = bbox2out([res], clsid2catid, is_bbox_normalized)

        if 'mask' in res:
            mask_results = mask2out([res], clsid2catid,
                                    model.mask_head.resolution)
        if 'landmark' in res:
            lmk_results = lmk2out([res], is_bbox_normalized)

        # visualize result
        im_ids = res['im_id'][0]
        # print(im_ids)
        for im_id in im_ids:
            image_path = imid2path[int(im_id)]
            image = Image.open(image_path).convert('RGB')
            # image1 = Image.open()

            # use VisualDL to log original image
            if FLAGS.use_vdl:
                original_image_np = np.array(image)
                # print(np.array(res['no_se']).shape)
                no_se = draw(np.array(outs[-5]),'new_featrues/no_se_{}'.format(vdl_image_frame), img_num = 10)
                is_se =  draw(np.array(outs[-4]),'new_featrues/is_se_{}'.format(vdl_image_frame))
                af_bb =  draw(np.array(outs[-3]),'new_featrues/af_bb_{}'.format(vdl_image_frame))
                no_se_1 =  draw(np.array(outs[-2]),'new_featrues/no_se_1_{}'.format(vdl_image_frame))
                is_se_1 =  draw(np.array(outs[-1]),'new_featrues/is_se_1_{}'.format(vdl_image_frame))
                # print('no_se',no_se)
                vdl_writer.add_image(
                    "original/frame_{}".format(vdl_image_frame),
                    original_image_np, vdl_image_step)
                
                vdl_writer.add_image(
                    "no_se/frame_{}".format(vdl_image_frame),
                    no_se, vdl_image_step)

                vdl_writer.add_image(
                    "is_se/frame_{}".format(vdl_image_frame),
                    is_se, vdl_image_step)

                vdl_writer.add_image(
                    "af_backbone/frame_{}".format(vdl_image_frame),
                    af_bb, vdl_image_step)

                vdl_writer.add_image(
                    "no_se_1/frame_{}".format(vdl_image_frame),
                    no_se_1, vdl_image_step)                 

                vdl_writer.add_image(
                    "is_se_1/frame_{}".format(vdl_image_frame),
                    is_se_1, vdl_image_step) 
            image = visualize_results(image,
                                      int(im_id), catid2name,
                                      FLAGS.draw_threshold, bbox_results,
                                      mask_results, lmk_results)

            # use VisualDL to log image with bbox
            if FLAGS.use_vdl:
                infer_image_np = np.array(image)
                vdl_writer.add_image("bbox/frame_{}".format(vdl_image_frame),
                                     infer_image_np, vdl_image_step)
                vdl_image_step += 1
                if vdl_image_step % 10 == 0:
                    vdl_image_step = 0
                    vdl_image_frame += 1

            save_name = get_save_image_name(FLAGS.output_dir, image_path)
            logger.info("Detection bbox results save in {}".format(save_name))
            image.save(save_name, quality=95)


if __name__ == '__main__':
    parser = ArgsParser()
    parser.add_argument(
        "--infer_dir",
        type=str,
        default=None,
        help="Directory for images to perform inference on.")
    parser.add_argument(
        "--infer_img",
        type=str,
        default=None,
        help="Image path, has higher priority over --infer_dir")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory for storing the output visualization files.")
    parser.add_argument(
        "--draw_threshold",
        type=float,
        default=0.5,
        help="Threshold to reserve the result for visualization.")
    parser.add_argument(
        "--use_vdl",
        type=bool,
        default=False,
        help="whether to record the data to VisualDL.")
    parser.add_argument(
        '--vdl_log_dir',
        type=str,
        default="vdl_log_dir/image",
        help='VisualDL logging directory for image.')
    FLAGS = parser.parse_args()
    main()
