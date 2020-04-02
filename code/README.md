# Code Sharing

1. Please download the code from "https://drive.google.com/open?id=14sKn_163FYqTVInV6OL_ZukT2wp2uPtn"
2. The folder contains the evaluation script for GQA-Attention model
3. Running the code with the following command

```
Running Command:
For Validation Balanced Set:

CUDA_VISIBLE_DEVICES=0 python run_experiments.py --do_val --id FinetuneTreeSparseStack2RemovalValSeed6777 --load_from model_ep0_0.5977  --model TreeSparsePostv2 --stacking 2

For Testdev Balanced Set:

CUDA_VISIBLE_DEVICES=0 python run_experiments.py --do_testdev_pred --id FinetuneTreeSparseStack2RemovalValSeed6777 --load_from model_ep0_0.5977  --model TreeSparsePostv2 --stacking 2

```

4. The trained model does not use the validation split provided by GQA dataset, therefore, the score is about 0.8% lower than reported in the paper.
5. If you have problem using the code, please shoot me an email.
