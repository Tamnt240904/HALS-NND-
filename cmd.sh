# # python preprocess.py
# Check accuracy first, then finetune if needed (2nd command).
# If finetuning is not desired, delete the "--load_classifier {finetuned_model}" in the commands.
python scripts/run_training.py --model resnet50 --layer layer3
python scripts/fine_tune.py --model resnet50 --layer layer3 --csae_model output/weights/original/resnet50_layer3_csae_masked_loss_model.pkl --epochs 3
python scripts/check_accuracy.py --model resnet50 --layer layer3 --csae_model output/weights/original/resnet50_layer3_csae_masked_loss_model.pkl --load_classifier output/weights/finetuned/resnet50_layer3_csae_masked_loss_finetuned.pth --num_samples 500

python scripts/run_training.py --model resnet101 --layer layer3
python scripts/fine_tune.py --model resnet101 --layer layer3 --csae_model output/weights/original/resnet101_layer3_csae_masked_loss_model.pkl --epochs 3
python scripts/check_accuracy.py --model resnet101 --layer layer3 --csae_model output/weights/original/resnet101_layer3_csae_masked_loss_model.pkl --load_classifier output/weights/finetuned/resnet101_layer3_csae_masked_loss_finetuned.pth --num_samples 500

python scripts/run_training.py --model vgg16 --layer features.23
python scripts/fine_tune.py --model vgg16 --layer features.23 --csae_model output/weights/original/vgg16_features_23_csae_masked_loss_model.pkl --epochs 3
python scripts/check_accuracy.py --model vgg16 --layer features.23 --csae_model output/weights/original/vgg16_features_23_csae_masked_loss_model.pkl --load_classifier output/weights/finetuned/vgg16_features_23_csae_masked_loss_finetuned.pth --num_samples 500

python scripts/run_training.py --model vgg19 --layer features.27
python scripts/fine_tune.py --model vgg19 --layer features.27 --csae_model output/weights/original/vgg19_features_27_csae_masked_loss_model.pkl --epochs 3
python scripts/check_accuracy.py --model vgg19 --layer features.27 --csae_model output/weights/original/vgg19_features_27_csae_masked_loss_model.pkl --load_classifier output/weights/finetuned/vgg19_features_27_csae_masked_loss_finetuned.pth --num_samples 500

python scripts/run_training.py --model densenet121 --layer features.denseblock3
python scripts/fine_tune.py --model densenet121 --layer features.denseblock3 --csae_model output/weights/original/densenet121_features_denseblock3_csae_masked_loss_model.pkl --epochs 3
python scripts/check_accuracy.py --model densenet121 --layer features.denseblock3 --csae_model output/weights/original/densenet121_features_denseblock3_csae_masked_loss_model.pkl --load_classifier output/weights/finetuned/densenet121_features_denseblock3_csae_masked_loss_finetuned.pth --num_samples 500

python scripts/run_training.py --model alexnet --layer features.10
python scripts/fine_tune.py --model alexnet --layer features.10 --csae_model output/weights/original/alexnet_features_10_csae_masked_loss_model.pkl --epochs 3
python scripts/check_accuracy.py --model alexnet --layer features.10 --csae_model output/weights/original/alexnet_features_10_csae_masked_loss_model.pkl --load_classifier output/weights/finetuned/alexnet_features_10_csae_masked_loss_finetuned.pth --num_samples 500

python scripts/run_training.py --model alexnet --layer features.6
python scripts/fine_tune.py --model alexnet --layer features.6 --csae_model output/weights/original/alexnet_features_6_csae_masked_loss_model.pkl --epochs 3
python scripts/check_accuracy.py --model alexnet --layer features.6 --csae_model output/weights/original/alexnet_features_6_csae_masked_loss_model.pkl --load_classifier output/weights/finetuned/alexnet_features_6_csae_masked_loss_finetuned.pth --num_samples 500

python scripts/run_training.py --model efficientnet_b4 --layer features.6
python scripts/fine_tune.py --model efficientnet_b4 --layer features.6 --csae_model output/weights/original/efficientnet_b4_features_6_csae_masked_loss_model.pkl --epochs 3
python scripts/check_accuracy.py --model efficientnet_b4 --layer features.6 --csae_model output/weights/original/efficientnet_b4_features_6_csae_masked_loss_model.pkl --load_classifier output/weights/finetuned/efficientnet_b4_features_6_csae_masked_loss_finetuned.pth --num_samples 500

python scripts/check_accuracy.py --all --num_samples 500 --mode finetuned # change mode here ['both', 'original', 'finetuned', 'smart']

# python visualize_multichannel_sae_resnet50.py --num_images 10
# python evaluate_fidelity.py --csae_model output/weights/multichannel_csae_resnet50_layer3_model.pkl
# python evaluate_fidelity.py --csae_model output/weights/multichannel_csae_vgg16_features_23_model.pkl --architecture vgg16 --target_layer features.23
# python evaluate_fidelity.py --csae_model output/weights/multichannel_csae_vgg19_features_27_model.pkl --architecture vgg19 --target_layer features.27
# python evaluate_fidelity.py --csae_model output/weights/multichannel_csae_densenet121_features_denseblock3_model.pkl --architecture densenet121 --target_layer features.denseblock3

# python scripts/visualize_features.py --model resnet50 --layer layer3 --csae_model output/weights/resnet50_layer3_csae_masked_loss_model.pkl --num_images 1