# # python preprocess.py
# python run_multichannel_csae.py --architecture resnet50 --target_layer layer3
# python run_multichannel_csae.py --architecture vgg16 --target_layer features.23
# python run_multichannel_csae.py --architecture vgg19 --target_layer features.27
# python run_multichannel_csae.py --architecture densenet121 --target_layer features.denseblock3
python run_multichannel_csae.py --architecture efficientnet_b0 --target_layer features.4
python run_multichannel_csae.py --architecture alexnet --target_layer features.6

# python visualize_multichannel_sae_resnet50.py --num_images 10
python evaluate_fidelity.py --csae_model output/weights/multichannel_csae_resnet50_layer3_model.pkl
python evaluate_fidelity.py --csae_model output/weights/multichannel_csae_vgg16_features_23_model.pkl --architecture vgg16 --target_layer features.23
python evaluate_fidelity.py --csae_model output/weights/multichannel_csae_vgg19_features_27_model.pkl --architecture vgg19 --target_layer features.27
python evaluate_fidelity.py --csae_model output/weights/multichannel_csae_densenet121_features_denseblock3_model.pkl --architecture densenet121 --target_layer features.denseblock3