_base_ = [
    "../_base_/models/faster_rcnn_r50_caffe_c4.py",
    "../_base_/datasets/voc0712.py",
    "../_base_/default_runtime.py",
]

# model settings
model = dict(roi_head=dict(bbox_head=dict(num_classes=20)))

# dataset settings
# use caffe img_norm
data_root = "data/VOCdevkit/"
img_norm_cfg = dict(mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", img_scale=(1000, 600), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(
    train=dict(
        dataset=dict(
            ann_file=data_root + "VOC2007/ImageSets/Main/trainval.txt",
            img_prefix=data_root + "VOC2007/",
            pipeline=train_pipeline,
        ),
    ),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline),
)

# schedule settings
# optimizer
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy="step", step=[3])
# runtime settings
total_epochs = 4  # actual epoch = 4 * 3 = 12
