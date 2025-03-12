# SMILE
The source code of "Dual-level Mixup for Graph Few-shot Learning with Fewer Tasks"

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


## Cite

If you find our work can help your research, please cite our work! <br>
```
@inproceedings{liu2025dual,
  title={Dual-level Mixup for Graph Few-shot Learning with Fewer Tasks},
  author={Liu, Yonghao and Li, Mengyu and Giunchiglia, Fausto and Huang, Lan and Li, Ximing and Feng, Xiaoyue and Guan, Renchu},
  booktitle={The Web Conference},
  year={2025}
}
```

## Contact
If you have any question, feel free to contact via [email](mailto:yonghao20@mails.jlu.edu.cn).
