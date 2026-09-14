import torch
import models_vit
import torch.nn as nn
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_

# call the model
model = models_vit.__dict__['RETFound_mae'](
    num_classes=2,
    drop_path_rate=0.2,
    global_pool=False,
)

root = 'model_weights/'

# load RETFound weights
checkpoint = torch.load(f'{root}/retfound_cfp.pt', map_location='cpu', weights_only=False)
checkpoint_model = checkpoint['model']
state_dict = model.state_dict()
for k in ['head.weight', 'head.bias']:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]

# # interpolate position embedding
interpolate_pos_embed(model, checkpoint_model)

# # load pre-trained model
msg = model.load_state_dict(checkpoint_model, strict=False)

# assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

# # manually initialize fc layer
trunc_normal_(model.head.weight, std=2e-5)
model.head = nn.Linear (1024, 5)
print("Model = %s" % str(model))
