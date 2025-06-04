
####Extract path-level features of training images
# imagenet_path is expected to be set as an environment variable
# The script will use hydra config for other parameters.
# Ensure 'data_dir' in conf/clip_feature_generation.yaml points to $imagenet_path or is overridden.
CUDA_VISIBLE_DEVICES=0 python clip_feature_generation.py

####Cluster the features to generate initialized codebook
# The script will use hydra config for parameters.
# Ensure 'feature_dir' in conf/minibatch_kmeans_per_class.yaml points to the output of clip_feature_generation.py
# and 'output_dir' points to the desired location for clustering centers.
CUDA_VISIBLE_DEVICES=0 python minibatch_kmeans_per_class.py