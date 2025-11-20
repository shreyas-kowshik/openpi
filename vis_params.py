import pickle
# data = 'logs/trainable_pi05_libero_lora_vision_full_ft_action_full_ft_siglip.pkl'
data = 'logs/trainable_pi05_libero_fullft_vision_lora_action_full_ft_siglip.pkl'
with open(data, 'rb') as f:
    data = pickle.load(f)
breakpoint()   