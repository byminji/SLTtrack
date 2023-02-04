# Tutorial for SLT-TransT & SLT-TrDiMP (Based on PyTracking)

SLT-TransT and SLT-TrDiMP are implemented based on PyTracking library, which is composed of [LTR](../ltr) for training and [pytracking](../pytracking) for testing.

## Training
For the detailed usage of the LTR library, please refer to [LTR](../ltr/README.md).

* Modify [ltr/admin/local.py](../ltr/admin/local.py) to set the paths to datasets, results, etc. 
* Download the baseline model (e.g., transt.pth) from [**[Models]**](https://drive.google.com/drive/folders/1gv7dIw6ywS47pjBkDWUrtWjdpjieyD6O?usp=sharing) and save it in your local path.
* (Optional) You can also train the baseline model by yourself by running the following command:
  ```
  python ltr/run_training.py transt transt
  ```
  
* Run the following commands to train the SLT tracker.
  Note that you should specify the path to the pretrained model in your training setting, e.g., [slt_transt.py](../ltr/train_settings/slt_transt/slt_transt.py).
  ```
  python ltr/run_training.py slt_transt slt_transt
  ```

## Testing

For the detailed usage of the pytracking library, please refer to [pytracking](../pytracking/README.md).

* Modify [pytracking/evaluation/local.py](../pytracking/evaluation/local.py) to set paths to datasets, results, etc.
* To test the tracker, run the following command:
  ```
  python pytracking/run_tracker.py [tracker_name] [parameter_name] --dataset_name [dataset_name]
  ```
  For example,
  ```
  python pytracking/run_tracker.py slt_transt slt_transt --dataset_name lasot
  ```
  * Tip: run with `--threads [num_threads]` and `--num_gpu [num_gpus]` for multi-gpu multi-threads inference.
  

* To see the evaluation results, run the following command:
  ```
  python pytracking/show_results.py [tracker_name] [parameter_name] --dataset_name [dataset_name]
  ```
  For example,
  ```
  python pytracking/show_results.py slt_transt slt_transt --dataset_name lasot
  ```
* To submit the results on evaluation servers (e.g., TrackingNet and GOT-10k), use the scripts in [util_scripts](../pytracking/util_scripts).