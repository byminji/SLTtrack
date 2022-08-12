# Tutorial for SLT-SiamRPN++ & SLT-SiamAttn (Based on PySOT)

SLT-SiamRPN++ and SLT-SiamAttn are implemented based on PySOT library, which is located in [pysot_toolkit](../pysot_toolkit).

## Data Preparation
If you want to train the baseline SiamRPN++ or SiamAttn by yourself, you should follow the guideline in [PySOT tutorial](https://github.com/STVIR/pysot/blob/master/TRAIN.md) to pre-process the datasets (cropping the original videos to make patches).

If you want to do SLT only, it is sufficient to download the dataset and specify the local path in your code.

## Training

* Modify [local_config.py](../pysot_toolkit/pysot/core/local_config.py) to set the paths to datasets, results, etc. 
* Download the baseline model (e.g., siamrpnpp.pth) from [**[Models]**](https://drive.google.com/drive/folders/1gv7dIw6ywS47pjBkDWUrtWjdpjieyD6O?usp=sharing) and save it in your local path.
* (Optional) You can also train the baseline model by yourself by running the following commands:
  ```
  cd pysot_toolkit
  
  # Specify GPU numbers for DDP training
  CUDA_VISIBLE_DEVICES=0,1,2,3
  
  # SiamRPN++
  python -m torch.distributed.run --nproc_per_node=4 train.py --expr siamrpnpp --cfg experiments/siamrpnpp/siamrpnpp.yaml
  
  # SiamAttn
  python -m torch.distributed.run --nproc_per_node=4 train_mask.py --expr siamattn --cfg experiments/siamattn/siamattn.yaml
  ```
  
* Run the following commands to train the SLT tracker. Note that you should specify the path to the pretrained model in your training setting, e.g., [slt_siamrpnpp.yaml](../pysot_toolkit/experiments/slt_siamrpnpp/slt_siamrpnpp.yaml).
  ```
  # SiamRPN++
  python train_slt.py --expr slt_siamrpnpp --cfg experiments/slt_siamrpnpp/slt_siamrpnpp.yaml
  
  # SiamAttn
  python train_slt_mask.py --expr slt_siamattn --cfg experiments/slt_siamattn/slt_siamattn.yaml
  ```


## Testing

* To test the tracker, run the following command:
  ```
  python test.py --expr [expr_name] --e [epoch] --dataset [dataset_name]
  ```
  For example,
  ```
  python test.py --expr slt_siamrpnpp --e 20 --dataset LaSOT
  ```
* The test results will be saved in the current directory (results/[dataset_name]/[expr_name]).
* To see the evaluation results, run the following command:
  ```
  python eval.py -p results --dataset [dataset_name]
  ```