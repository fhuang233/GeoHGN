# Geometric Heterogeneous Graph Neural Network for Protein-Ligand Binding Affinity Prediction

A Python module for protein-ligand binding affinity prediction

## Requirements

This project relies on specific Python packages to ensure its proper functioning. The required packages and their versions are listed in the `requirements.txt` file.

## Data

`./data.zip` contains the processed graph data and splits in PDB benchmark. './LBAdata.zip' contains the processed graph data and its splits in LBA benchmark. './cold_start_data' contains the splits for evaluation in cold start scenario


Training for the GeoHGN model with the dataset `./data.zip` for PDB benchmark

```
unzip data.zip
python main.py --lr 0.0001 --wd 5e-6 --runs 3 --batch_size 128 --cutoff_inter 5.0 --cutoff_intra 6.0 --num_conv_layers 6 --num_radial_intra 3 --num_spherical_intra 2 --num_radial_inter 3 --num_spherical_inter 2 --num_res_layers 4 --num_trans_layers 4 --num_output_layers 4 --num_dist_rbf_intra 16 --num_dist_rbf_inter 16 --num_heads 8 --hidden_channels 256 --cuda 0 --early_stop_epoch 200 --epochs 1000 
```

Training for the GeoHGN model with the dataset `./LBAdata.zip` for LBA benchmark

```
unzip LBAdata.zip
python main_LBA.py --lr 0.0001 --wd 5e-6 --runs 3 --batch_size 128 --cutoff_inter 5.0 --cutoff_intra 6.0 --num_conv_layers 3 --num_radial_intra 3 --num_spherical_intra 2 --num_radial_inter 3 --num_spherical_inter 2 --num_res_layers 4 --num_trans_layers 0 --num_output_layers 4 --num_dist_rbf_intra 16 --num_dist_rbf_inter 16 --num_heads 8 --hidden_channels 256 --cuda 0 --early_stop_epoch 200 --epochs 1000 

```

Training for the GeoHGN model for cold start study

```
unzip data.zip
python main_cold_start.py --lr 0.0001 --wd 5e-6 --runs 3 --batch_size 128 --cutoff_inter 5.0 --cutoff_intra 6.0 --num_conv_layers 4 --num_radial_intra 4 --num_spherical_intra 2 --num_radial_inter 4 --num_spherical_inter 2 --num_res_layers 3 --num_trans_layers 4 --num_output_layers 4 --num_dist_rbf_intra 32 --num_dist_rbf_inter 32 --num_heads 8 --hidden_channels 256 --cuda 0 --early_stop_epoch 200 --epochs 1000 
```

