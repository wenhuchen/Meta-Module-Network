# Evaluation Code Sharing

1. Please download the code from "https://drive.google.com/open?id=14sKn_163FYqTVInV6OL_ZukT2wp2uPtn"
2. The folder contains the evaluation script for GQA-Attention model
3. Running the code with the following command

Running Command:
For Validation Balanced Set:
```
CUDA_VISIBLE_DEVICES=0 python run_experiments.py --do_val --id FinetuneTreeSparseStack2RemovalValSeed6777 --load_from model_ep0_0.5977  --model TreeSparsePostv2 --stacking 2
```
For Testdev Balanced Set:
```
CUDA_VISIBLE_DEVICES=0 python run_experiments.py --do_testdev_pred --id FinetuneTreeSparseStack2RemovalValSeed6777 --load_from model_ep0_0.5977  --model TreeSparsePostv2 --stacking 2

```

4. The trained model does not use the validation split provided by GQA dataset, therefore, the score is about 0.8% lower than reported in the paper.
5. If you have problem using the code, please shoot me an email.

# Training Code
Since the Bottom-Up Feature for training is too large to share, therefore, we won't be able to provide the used full training feature. However, you provide the code for reproceducing the trained model:
  1. Bootstrapping
  ```
  python run_experiments.py --do_train_all --model TreeSparsePostv2 --id TreeSparsePost2FullValidationRemoved --stacking 2 --batch_size 1024
  ```
  2. Finetunning, loading the bootstrapped model from the corresponding folder.
  ```
  python run_experiments.py --do_finetune --id FinetuneTreeSparseStack2RemovalValSeed6888 --model TreeSparsePostv2 --load_from models/TreeSparsePost2FullValidationRemoved/model_ep2_0.5587 --seed 6888 --stacking 2
  ```
