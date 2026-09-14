import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple, Dict
from survival_on_embedding.model import (
    BreslowEstimator,
    predict_survival, 
    get_ibs_sksurv,
    get_brier_scores_sksurv,
    get_aurocs_sksurv,
    get_concordance_index_sksurv,
    augur_predict_survival
) 

from survival_on_embedding.utils import Logger


def Evaluate(
    survival_model: torch.nn.Module,
    encoder: torch.nn.Module,
    breslow: BreslowEstimator,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    device: str,
    logger: Logger,
    epoch: int = None,
    train_data=None,
    testrun=None,
    return_logits = False,
    ssl_method = None,
    data_name = '',
    evaluation_visits_from_now = [2, 4, 6, 8, 10],    
    use_r = False,
    return_curves = False,
    return_all = False,
) -> Tuple[dict, dict]:
    performance_dict = {}
    out_dict = {}

    print('in evaluate considering eval time', evaluation_visits_from_now)
    for split, _ in dataloaders.items():
        
        performance = {}
        logits = []
        _events = []
        _durations = []
        _paths = []

        survival_model.eval()
        encoder.eval()

        all_images = []
        print(f"evaluating for times {evaluation_visits_from_now}")
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloaders[split], desc="Evaluation Batches")):  
                if data_name == 'augur':                  
                    imgs, grades, events, durations, metadata, paths, eye_ids = batch
                else:
                    imgs, grades, events, durations, paths = batch
                _events.extend(events)
                _durations.extend(durations)

                imgs = imgs.to(device)
                events, durations = events.to(device), durations.to(device)

                # Get basenames
                if data_name != 'augur':
                
                    paths = [p.split("/")[-1] for p in paths]
                
                _paths.extend(paths)

                # Get embeddings
             
                embeddings = encoder(imgs)
                # all_images.extend(imgs)

                if ssl_method == 'swin':
                    if embeddings.dim() == 3:         # (B, L, C)
                        embeddings = embeddings.mean(dim=1)   
                    elif embeddings.dim() == 4:       # (B, C, H, W)
                        embeddings = embeddings.mean(dim=(1, 2))
              
                log_partial_hazards = survival_model(embeddings)
                logits.extend(log_partial_hazards)

                if testrun and i > 10:
                    break

        # Compute performance metrics
        assert breslow is not None, "Need to pass fitted Breslow object."
        assert train_data is not None, "Need to pass training data for Breslow estimation."

        l = torch.tensor(logits).detach().cpu()
        print('getting logits shape', l.shape)

        if data_name != 'augur':
            survival_curves = predict_survival(breslow, l, evaluation_visits_from_now)
        else:
            survival_curves, time_points = augur_predict_survival(breslow, l, evaluation_visits_from_now)
            # survival_curves, time_points = augur_predict_survival(breslow, torch.tensor(_logits), evaluation_visits_from_now)
            mask = ~np.isnan(time_points)
            time_points = time_points[mask]
            survival_curves = survival_curves[:, mask]
            
        if return_all:
            return survival_curves, np.array(_events), np.array(_durations), _paths, l

        if return_curves:
            return survival_curves, np.array(_events), np.array(_durations)

        print(f"Evaluating split {split} with {len(_events)} samples. \n ")
        
        performance["ibs"] = get_ibs_sksurv(
            train_data, _events, _durations, survival_curves, evaluation_visits_from_now
        )
        performance["brier_scores"] = get_brier_scores_sksurv(
            train_data, _events, _durations, survival_curves, evaluation_visits_from_now
        )
        performance["aurocs"] = get_aurocs_sksurv(
            train_data, _events, _durations, survival_curves, evaluation_visits_from_now
        )
        performance["mean_auroc"] = np.mean(performance["aurocs"])

        performance["concordance_index"] = get_concordance_index_sksurv(
            _events, _durations, survival_curves, evaluation_visits_from_now
        )
        
        out_df = pd.DataFrame(
            {
                "survival_curve": list(survival_curves),
                "event": _events,
                "duration": _durations,
                "image_name": _paths,
            }
        )

        logger.log(performance, step=epoch, split=split)
        logger.log({
            f"{split}/brier_score_t{evaluation_visits_from_now[i]}": v
            for i, v in enumerate(performance["brier_scores"])
        }, step=epoch)
        logger.log_survival_curves(out_df, split)

        performance_dict[split] = performance
        out_dict[split] = out_df

    if return_logits:
        return out_dict, performance_dict, l, survival_curves

    return out_dict, performance_dict
        # else:
        #     return survival_curves, _events, _durations
            

def EvaluateAugur(
    survival_model: torch.nn.Module,
    encoder: torch.nn.Module,
    breslow: BreslowEstimator,
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    device: str,
    logger: Logger,
    epoch: int = None,
    train_data=None,
    testrun=None,
    return_logits = False,
    ssl_method = None,
    data_name = '',
    evaluation_visits_from_now = None,
    use_r = False,
    return_curves = True,
) -> Tuple[dict, dict]:
    performance_dict = {}
    out_dict = {}

    for split, _ in dataloaders.items():
        print('in split', split)
        performance = {}
        logits = []
        _events = []
        _durations = []
        _paths = []

        survival_model.eval()
        encoder.eval()

        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloaders[split], desc="Evaluation Batches")):
                if (data_name == 'augur'):    
                    imgs, grades, events, durations, metadata, paths, eye_ids = batch
                else:
                    imgs, grades, events, durations, paths = batch
                _events.extend(events)
                _durations.extend(durations)

                imgs = imgs.to(device)
                events, durations = events.to(device), durations.to(device)

                # Get basenames
                if data_name != 'augur':
                
                    paths = [p.split("/")[-1] for p in paths]
                
                _paths.extend(paths)

                # Get embeddings
             
                embeddings = encoder(imgs)

                if ssl_method == 'swin':
                    if embeddings.dim() == 3:         # (B, L, C)
                        embeddings = embeddings.mean(dim=1)   
                    elif embeddings.dim() == 4:       # (B, C, H, W)
                        embeddings = embeddings.mean(dim=(1, 2))
              
                log_partial_hazards = survival_model(embeddings)
                logits.extend(log_partial_hazards)

                if testrun and i > 10:
                    break

        # Compute performance metrics
        assert breslow is not None, "Need to pass fitted Breslow object."
        assert train_data is not None, "Need to pass training data for Breslow estimation."

        l = torch.tensor(logits).detach().cpu()
        if split == 'val':
            evaluation_visits_from_now = [2, 4, 6, 8, 10]
            survival_curves = predict_survival(breslow, l, evaluation_visits_from_now)
        else:
            survival_curves = predict_survival(breslow, l, evaluation_visits_from_now)
            # survival_curves, time_points = augur_predict_survival(breslow, torch.tensor(_logits), evaluation_visits_from_now)
            # mask = ~np.isnan(time_points)
            # time_points = time_points[mask]
            # survival_curves = survival_curves[:, mask]

    if return_curves == True:
        print('in first condition')
        return survival_curves, _events, _durations
        
    elif ((use_r == False) and (return_curves == False)):
        print('in second')
        print(f"Evaluating split {split} with {len(_events)} samples. \n ")
        
        performance["ibs"] = get_ibs_sksurv(
            train_data, _events, _durations, survival_curves, evaluation_visits_from_now
        )
        performance["brier_scores"] = get_brier_scores_sksurv(
            train_data, _events, _durations, survival_curves, evaluation_visits_from_now
        )
        performance["aurocs"] = get_aurocs_sksurv(
            train_data, _events, _durations, survival_curves, evaluation_visits_from_now
        )
        performance["mean_auroc"] = np.mean(performance["aurocs"])

        performance["concordance_index"] = get_concordance_index_sksurv(
            _events, _durations, survival_curves, evaluation_visits_from_now
        )

        print(f"Done. Evaluated ibs score: {performance['ibs']}")

        out_df = pd.DataFrame(
            {
                "survival_curve": list(survival_curves),
                "event": _events,
                "duration": _durations,
                "image_name": _paths,
            }
        )

        logger.log(performance, step=epoch, split=split)
        logger.log_survival_curves(out_df, split)

        performance_dict[split] = performance
        out_dict[split] = out_df
        return out_dict, performance_dict, 

    # else:
    #     print('in third')
    #     from survival_on_embedding.model import RMetrics
    #     survival_curves = torch.tensor(np.array(survival_curves))
    #     _events = torch.tensor(np.array(_events))
    #     _durations = torch.tensor(np.array(_durations))

    #     rmetrics = RMetrics(preds=survival_curves, events=_events, evaluation_visits_from_now = _durations)
    #     tpr, fpr = rmetrics.calc_rocs_timeroc()
    #     print('\n \n \n **** metrics after split in wandb *** \n \n \n', tpr)
    #     print('\n \n \n **** metrics after split in wandb *** \n \n \n', fpr)

    #     return tpr, fpr
        
        # return survival_curves, _events, _durations
        
    #     print(f"Evaluating split {split} with {len(_events)} samples. \n ")
        
    #     performance["ibs"] = get_ibs_sksurv(
    #         train_data, _events, _durations, survival_curves, evaluation_visits_from_now
    #     )
    #     performance["brier_scores"] = get_brier_scores_sksurv(
    #         train_data, _events, _durations, survival_curves, evaluation_visits_from_now
    #     )
    #     performance["aurocs"] = get_aurocs_sksurv(
    #         train_data, _events, _durations, survival_curves, evaluation_visits_from_now
    #     )
    #     performance["mean_auroc"] = np.mean(performance["aurocs"])

    #     performance["concordance_index"] = get_concordance_index_sksurv(
    #         _events, _durations, survival_curves, evaluation_visits_from_now
    #     )

    #     print(f"Done. Evaluated ibs score: {performance['ibs']}")

    #     out_df = pd.DataFrame(
    #         {
    #             "survival_curve": list(survival_curves),
    #             "event": _events,
    #             "duration": _durations,
    #             "image_name": _paths,
    #         }
    #     )

    #     logger.log(performance, step=epoch, split=split)
    #     logger.log_survival_curves(out_df, split)

    #     performance_dict[split] = performance
    #     out_dict[split] = out_df

    # from survival_on_embedding.model import RMetrics
    # rmetrics = RMetrics(survival_curves, time_points)
    # tpr, fpr = rmetrics.calc_rocs_timeroc()