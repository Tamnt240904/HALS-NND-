# # python preprocess.py
python scripts/run_training.py --model resnet50 --layer layer3 
python scripts/run_training.py --model resnet101 --layer layer3
python scripts/run_training.py --model vgg16 --layer features.23
python scripts/run_training.py --model vgg19 --layer features.27
python scripts/run_training.py --model densenet121 --layer features.denseblock3
python scripts/run_training.py --model alexnet --layer features.10
python scripts/run_training.py --model alexnet --layer features.6
python scripts/run_training.py --model efficientnet_b4 --layer features.6

python scripts/check_accuracy.py --model alexnet --layer features.6 --csae_model output/weights/alexnet_features_6_csae_masked_loss_model.pkl --num_samples 500

python scripts/check_accuracy.py --model densenet121 --layer features.denseblock3 --csae_model output/weights/densenet121_features_denseblock3_csae_masked_loss_model.pkl --num_samples 500


python scripts/check_accuracy.py --all --num_samples 500

# python visualize_multichannel_sae_resnet50.py --num_images 10
# python evaluate_fidelity.py --csae_model output/weights/multichannel_csae_resnet50_layer3_model.pkl
# python evaluate_fidelity.py --csae_model output/weights/multichannel_csae_vgg16_features_23_model.pkl --architecture vgg16 --target_layer features.23
# python evaluate_fidelity.py --csae_model output/weights/multichannel_csae_vgg19_features_27_model.pkl --architecture vgg19 --target_layer features.27
# python evaluate_fidelity.py --csae_model output/weights/multichannel_csae_densenet121_features_denseblock3_model.pkl --architecture densenet121 --target_layer features.denseblock3

python scripts/visualize_features.py --model resnet50 --layer layer3 --csae_model output/weights/resnet50_layer3_csae_masked_loss_model.pkl --num_images 1
