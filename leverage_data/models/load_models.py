from .surv_models import *
from .dino import build_model_from_cfg
from pathlib import Path
from omegaconf import OmegaConf

def get_encoder(weights_path:str=None,  device:str="cuda:0",
                 img_size = 224, eval_augur = False, cloud = 'mlcloud', args = None,
                   config_file = None):
    
    embed_dim = 512
    backbone_model = None
    if (weights_path is None) or ('scratch' in weights_path):
        print("training from scratch")
        backbone = resnet18_modified() #resnet18_pytorch
        weights_path = "scratch"
        backbone_model = "resnet18_scratch"
        

    elif "resnet_imagenet" in weights_path.lower():
        print("using imagenet")
        backbone = resnet18_modified_pretrained()
        backbone_model = "resnet_imagenet" # changed from imagenet_resnet18_pytorch to imagenet on 21-02-26

    elif "simclr" in weights_path.lower():
        print('in simclr')
        model_str = "_".join(weights_path.split('arc')[-1].split('checkpoint')[0].strip('_').strip('/').split('_')[:-1])
        print('initial model str', model_str)
        if 'TMI' in model_str:
            model_str = '_'.join(model_str.split('_')[:-1])
            if cloud != 'mlcloud':
                model_str = model_str.split('TMI')[0]
            print('updated model_str', model_str)
        else:
            model_str = weights_path.split('arc')[-1].split('checkpoint')[0].strip('_').split('RC')[0]
            if 'sim' in model_str:
                model_str = "_".join(model_str.split("_")[:-3])     

        model_str = model_str.strip("_")
        print('model_str', model_str)
        main_backbone = RESNET_MODEL_DICT[model_str]
        ssl_model = SimCLR(main_backbone)

        checkpoint = torch.load(weights_path,  weights_only=False)        
        ssl_model.load_state_dict(checkpoint['model_weights'], strict=True)

        backbone = ssl_model.backbone
        if 'resnet18_modified_pretrained' in weights_path:
            backbone_model = 'simclr_inet_nako'
        elif 'resnet18_modified_TMI_simclr' in weights_path:
            backbone_model = 'simclr_nako'
        else:
            raise ValueError('Please check that the name of the simclr model is correct')

    elif 'arc_mae_vit_base_patch16' in weights_path:# or \
        model, embed_dim = mae_nako_weights(weights_path)
        
        backbone = ModelWithMAE(model, use_cls= True)
        backbone_model = "nako_mae"
        
    elif 'retfound' in weights_path.lower():
        if isinstance(img_size, tuple):
            img_size = img_size[0]
        assert img_size == 224, 'please ensure you are using the right image size'
        print('retfound img_size', img_size)
        backbone, embed_dim = retfound_encoder(img_size, weights_path)
        backbone_model = "retfound_mae_vit"
        # print(backbone)


    elif 'metad_pretrained' in weights_path.lower():
        print('in ssl_encoder metad pretrained')
        # dino_weights_pretrained = bool(args.dino_weights_pretrained)
        dino_model = load_dino_pretrained(eval_augur)
        print('dino pretrained model loaded')
        backbone  = ViTBackbone(dino_model, use_cls=True)
        embed_dim = backbone.feat_dim
        backbone_model = 'dino_pretrained'


    elif 'dino_nako' in weights_path.lower():
        config_file = str(Path(weights_path).resolve().parent)+'/config_updated.yaml'
        cfg = OmegaConf.load(config_file)
        _, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        checkpoints = torch.load(weights_path, map_location= 'cpu')
        backbone_state_dict = {k.replace('backbone.', ''):v
                            for k, v in  checkpoints['teacher'].items() 
                            if k.startswith('backbone.')}
        teacher_backbone.load_state_dict(backbone_state_dict, strict=True)
        # dino_model, embed_dim = load_dino_nako(weights_path, config_file)
        backbone  = DINOBackbone(teacher_backbone, use_cls=True)
        backbone_model = 'dino_nako'
        print('dino_patch_size',teacher_backbone.patch_embed.patch_size) 
        print('num patch_embed',teacher_backbone.patch_embed.num_patches) 

    elif 'dinov2_vitb14_pretrain' in weights_path.lower():
        print('in dinov2_vitb14_pretrain ')
        config_file = 'sssl_default_config.yaml'
        cfg = OmegaConf.load(config_file)
        _, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        dino_model, embed_dim = load_dinov2_weights(teacher_backbone,
                                                     weights_path)
        backbone  = DINOBackbone(dino_model, use_cls=True)
        backbone_model = 'dino_pretrained_224'
        print('dino_patch_size',dino_model.patch_embed.patch_size) 
        print('num patch_embed',dino_model.patch_embed.num_patches)
        
    else:
        print(f"\n \n ***UNKNOWN {weights_path}*** \n \n")
        return ValueError("Unknown encoder")
    return backbone,  embed_dim, weights_path, backbone_model


