# python training/train.py --use_context_encoder=False --model_type=MLP
# python training/train.py --model_type=rwm --action_type=torque
# # python training/train.py --model_type=rwm --action_type=policy
# python training/train.py --use_context_encoder=False --delan_use_film=False --model_type=whole --delan_use_history=False
python training/train.py --use_context_encoder=False \
    --delan_use_film=False --model_type=whole --delan_use_history=True


# for a in {41..60}; 
# do
# python eval/animation.py --episode_index=$a
# done