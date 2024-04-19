python main.py \
--root /datasets \
--desc protosim_seed0 \
--lr 5e-5 \
--optim sgd \
--weight_decay 0.0005 \
--project_name Protosim \
--proto_optim sgd \
--seed 0 \
--sim_loss_lr 0.01 \
--sim_alpha 10000.0 \
--center_alpha 0.0001  \
--center_loss_lr 0.0001 \
--focalloss \
--sigma 0.3 \

