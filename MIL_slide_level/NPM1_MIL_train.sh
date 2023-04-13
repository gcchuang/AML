# # run python to train the MIL model with NPML1 dataset, output the log file to NPM1_output.log
# python -u MIL_train.py --output /home/weber50432/AML_image_processing/MIL_slide_level/output/NPM1_batch_512 --train_lib /home/weber50432/AML_image_processing/lib/NPM1/NPM1_train_data.pt --val_lib /home/weber50432/AML_image_processing/lib/NPM1/NPM1_val_data.pt --batch_size 512 --nepochs 100 --workers 1 > /home/weber50432/AML_image_processing/MIL_slide_level/output/NPM1_batch_512/train_output.log 2>&1 &


# # run python to train the MIL model with NPML1 dataset, output the log file to NPM1_output.log
# python -u MIL_train.py --output /home/weber50432/AML_image_processing/MIL_slide_level/output/NPM1_batch_1024 --train_lib /home/weber50432/AML_image_processing/lib/NPM1/NPM1_train_data.pt --val_lib /home/weber50432/AML_image_processing/lib/NPM1/NPM1_val_data.pt --batch_size 1024 --nepochs 100 --workers 1 --test_every 20 --k 3 > /home/weber50432/AML_image_processing/MIL_slide_level/output/NPM1_batch_1024/train_output.log 2>&1 &

# run python to train the MIL model with NPML1 dataset, output the log file to NPM1_output.log
python -u MIL_train.py --output /home/weber50432/AML_image_processing/MIL_slide_level/output/NPM1_batch_128 --train_lib /home/weber50432/AML_image_processing/lib/NPM1/NPM1_train_data.pt --val_lib /home/weber50432/AML_image_processing/lib/NPM1/NPM1_val_data.pt --batch_size 128 --nepochs 100 --workers 1 > /home/weber50432/AML_image_processing/MIL_slide_level/output/NPM1_batch_128/train_output.log 2>&1 &