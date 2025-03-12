# SMILE
The anonymous submission of source code for "A Simple Yet Effective Approach for Graph Few-Shot Learning with Fewer Tasks"

Here we provide three datasets for reproducibility, including ``Amazon-Clothing``, ``CoraFull``, and ``DBLP``, with two datasets in the zip file and the other available for automatic download.
## Usage
You can run the following the command.
```
cd SMILE
```

```
unzip few_shot_data
```

For in-domain setting
```
python train.py --dataset Amazon_clothing --way 5 --shot 5 --num_tasks 5
```
You can change the ```--dataset``` to ```corafull``` or ```dblp``` to train other datasets.

For cross-domain setting
```
python train_cross_domain.py --dataset Amazon_clothing --dataset_cr corafull
```

You can change the ```--dataset``` to ```corafull``` and the ```--dataset_cr``` to ```Amazon_clothing``` to switch the cross-domain setting.