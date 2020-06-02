# CMND_space_detect
Detect space in an image containing 2 words

## Usage
```
git clone https://github.com/quoccuonglqd/CMND_space_detect.git
cd CMND_space_detect
python Detect_space.py --mode $MODE --image_path $path_to_image --folder_path $path_to_directory --config_path $path_to_config_file
```

Note: mode argument can take value in {single,multiple},which represent for processing on a single image or multiple images. image_path 
need to be fed if mode is single. Otherwise, folder_path is needed.
