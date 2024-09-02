input_dir_abs=../../image_classification
src_out_abs=./
hdr_out_abs=./
img_size=128

python gen_rgb_cpp.py \
    --image_path ${input_dir_abs} \
    --source_folder_path ${src_out_abs} \
    --header_folder_path ${hdr_out_abs} \
    --image_size ${img_size} ${img_size}

python gen_rgb_bin.py \
    --image_path ${input_dir_abs} \
    --source_folder_path ${src_out_abs} \
    --header_folder_path ${hdr_out_abs} \
    --image_size ${img_size} ${img_size}

mv *.cc ../
mv *.hpp ../
mv *.bin ../
