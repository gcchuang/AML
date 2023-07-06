output_path=./output/NPM1_cell_test
lib_path=/home/weber50432/AML_image_processing/lib/NPM1_cell_test
patches_path=/home/exon_storage1/aml_slide/single_cell_image/
python -u MIL_train_cell.py --output $output_path --train_lib ${lib_path}/NPM1_train_data.pt --val_lib ${lib_path}/NPM1_val_data.pt --patches_path $patches_path --batch_size 128 --nepochs 50 --workers 1 --k 20 > ${output_path}/train_output.log 2>&1 &
