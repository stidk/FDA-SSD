# 简介

该模型为FDA-SSD模型代码,采用的是PaddlePaddle1.8.0,python3.7.0.在ai studio平台上运行.voc数据集为ai studio平台公开的由笨笨用户上传的pascal-voc数据集,遥感数据集为ai studio平台公开的由AIStudio322801用户上传的遥感数据数据集.

# 解压数据集
```Shell
!unzip -q /home/aistudio/data/data4379/pascalvoc.zip -d /home/aistudio/FDA-SSD/dataset/voc
!python FDA-SSD/dataset/voc/create_list.py -d 'FDA-SSD/dataset/voc/pascalvoc'
```
其中,data4379为上传数据集的编号,可能不一样,可以在data目录下查看.

# 运行
```Shell
%cd ~/FDA-SSD
!python tools/train.py -c ssd_r50_300_voc.yml --eval
```
若要运行其它代码,可以将ssd_r50_300_voc.yml改成主界面有的所有yml文件,hrrsd结尾文件需更改ppdet/data/source/voc.py文件的104行中的pascalvoc_label改为pascalvoc_label1即可.
图像大小为300的voc数据集平均每个模型训练时间为10小时左右,图像大小为512的voc数据集平均每个模型训练时间为24左右,hrrsd数据集的平均运行时间在30小时左右.
