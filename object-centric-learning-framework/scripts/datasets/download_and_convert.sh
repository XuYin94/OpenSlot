#!/bin/bash

case $1 in
  COCO)
    echo "Downloading COCO2017 and COCO20k data to data/coco"
    # ./download_scripts/download_coco_data.sh

    echo "Converting COCO2017 to webdataset stored at outputs/coco2017"
    SEED=23894734
    mkdir -p outputs/coco2017/train
    python conversion_scripts/convert_coco.py data/coco/train2017 outputs/coco2017/train --instance data/coco/annotations/instances_train2017.json --stuff data/coco/annotations/stuff_train2017.json --caption data/coco/annotations/captions_train2017.json --seed $SEED
    mkdir -p outputs/coco2017/val
    python conversion_scripts/convert_coco.py data/coco/val2017 outputs/coco2017/val --instance data/coco/annotations/instances_val2017.json --stuff data/coco/annotations/stuff_val2017.json --caption data/coco/annotations/captions_val2017.json --seed $SEED
    mkdir -p outputs/coco2017/test
    python conversion_scripts/convert_coco.py data/coco/test2017 outputs/coco2017/test --test data/coco/annotations/image_info_test2017.json --seed $SEED
    mkdir -p outputs/coco2017/unlabeled
    python conversion_scripts/convert_coco.py data/coco/unlabeled2017 outputs/coco2017/unlabeled --test data/coco/annotations/image_info_unlabeled2017.json --seed $SEED

    echo "Converting COCO20k to webdataset stored at outputs/coco2014/20k"
    mkdir -p outputs/coco2014/20k
    python conversion_scripts/convert_coco.py data/coco/train2014 outputs/coco2014/20k --instance data/coco/annotations/instances_train2014.json --caption data/coco/annotations/captions_train2014.json --seed $SEED --subset_list misc/coco20k_files.txt
    ;;


  voc2007)
    echo "Creating voc2007 webdataset in outputs/voc2007"
    # Ensure downloaded data is stored in data folder.
    export TFDS_DATA_DIR=data/tensorflow_datasets
    mkdir -p outputs/voc2007/train
    python conversion_scripts/convert_tfds.py extended_voc/2007-segmentation train outputs/voc2007/train
    mkdir -p outputs/voc2007/val
    python conversion_scripts/convert_tfds.py extended_voc/2007-segmentation validation outputs/voc2007/val
    mkdir -p outputs/voc2007/test
    python conversion_scripts/convert_tfds.py extended_voc/2007-segmentation test outputs/voc2007/test
    ;;


  voc2012)
    # Augmented pascal voc dataset with segmentations and additional instances.
    echo "Creating voc2012 webdataset in outputs/voc2012"
    # Ensure downloaded data is stored in data folder.
    export TFDS_DATA_DIR=data/tensorflow_datasets
    mkdir -p outputs/voc2012/trainaug
    python conversion_scripts/convert_tfds.py extended_voc/2012-segmentation train+sbd_train+sbd_validation outputs/voc2012/trainaug
    # Regular pascal voc splits.
    mkdir -p outputs/voc2012/train
    python conversion_scripts/convert_tfds.py extended_voc/2012-segmentation train outputs/voc2012/train
    mkdir -p outputs/voc2012/val
    python conversion_scripts/convert_tfds.py extended_voc/2012-segmentation validation outputs/voc2012/val
    ;;




  *)
    echo "Unknown dataset $1"
    echo "Only COCO, clevr+cater, clevrer, voc2007, voc2012, movi_c and movi_e are supported."
    ;;
esac
