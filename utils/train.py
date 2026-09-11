from typing import Tuple, Dict
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import time
import copy
import os
torch.multiprocessing.set_sharing_strategy('file_system') # Avoid "too many open files" error

from survival_on_embedding.model import (
    BreslowEstimator,
    fit_breslow,
    predict_survival,
    get_survival_loss,
    augur_predict_survival

)
from survival_on_embedding.predict.surv_eval import Evaluate
from survival_on_embedding.utils import Logger


def mmd_loss(xs: torch.Tensor, yt: torch.Tensor) -> torch.Tensor:
    mean_xs = xs.mean(dim=0)
    mean_yt = yt.mean(dim=0)
    return torch.norm(mean_xs - mean_yt, p=2) ** 2

def train_one_epoch(
    survival_model: torch.nn.Module,
    encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    device: str,
    logger: Logger,
    epoch: int,
    testrun = None,
    ssl_method = None,
    tcn_loss = None,
    data_name = '',
    evaluation_visits_from_now = [2, 4, 6, 8, 10],
    args = None


) -> Tuple[BreslowEstimator, np.ndarray, float]:
    loss_logging = []
    _logits = []
    _events = []
    _durations = [] 

    # accelerator = Accelerator()
    survival_model.train()

    if args.freeze_encoder:
        print(f"freeze encoder is {args.freeze_encoder}, so encoder set to evaluate")
        encoder.eval()
    else:
        print(f"freeze encoder is {args.freeze_encoder}, so encoder set to train")
        encoder.train()
    

#     encoder, survival_model, optimizer, train_loader = accelerator.prepare(
#     encoder, survival_model, optimizer, train_loader
# )
    # pre_session_time = time.time()
    for i, (imgs, grades, events, durations, paths, *_) in enumerate(
        tqdm(train_loader, desc="Training Batches")
    ):
        start_time = time.time()
        _events.extend(events)
        _durations.extend(durations)
        
        data_time = time.time()
        imgs = imgs.to(device)
        events, durations = events.to(device), durations.to(device)
        

        # Get basenames
        # paths = [p.split("/")[-1] for p in paths]

        # Get embeddings
        
        embeddings = encoder(imgs)

        for opt in optimizer:
            opt.zero_grad()

        if ssl_method == 'swin':
            if embeddings.dim() == 3:         # (B, L, C)
                embeddings = embeddings.mean(dim=1)   
            elif embeddings.dim() == 4:       # (B, C, H, W)
                embeddings = embeddings.mean(dim=(1, 2))

        log_partial_hazards = survival_model(embeddings) #pass embeddings to MLP
        _logits.extend(log_partial_hazards)

        loss = get_survival_loss(log_partial_hazards, events, durations, device)
        # forward_time = time.time()
        if tcn_loss is not None:
            t_loss = tcn_loss + loss
            loss_logging.append(t_loss.item())

            t_loss.backward()
        else:
            loss_logging.append(loss.item())
            loss.backward()
            # accelerator.backward(loss)  # 
        for opt in optimizer:
            opt.step()

        if testrun and i > 10:
            break
        # neural_end_time = time.time()

    # Fit Breslow estimator
    l = torch.tensor(_logits)
    breslow = fit_breslow(l, torch.tensor(_durations), torch.tensor(_events))

    if data_name != 'augur':
        survival_curves = predict_survival(breslow, torch.tensor(_logits), evaluation_visits_from_now)
    else:
        # print('evaluation_visits_from_now, evaluation_visits_from_now', evaluation_visits_from_now)
        survival_curves, time_points = augur_predict_survival(breslow, torch.tensor(_logits), evaluation_visits_from_now)
        mask = ~np.isnan(time_points)
        time_points = time_points[mask]
        survival_curves = survival_curves[:, mask]

    avg_loss = np.mean(loss_logging)
    return breslow, survival_curves, avg_loss


def train_survival_model(
    survival_model: torch.nn.Module,
    encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    start_epoch: int,
    epochs: int,
    early_stopping_patience: int,
    logger: Logger,
    train_data=None,
    testrun = None,
    ssl_method = None,
    tcn_loss = None,
    data_name = '',
    evaluation_visits_from_now = [2, 4, 6, 8, 10],
    sweep = None,
    train_size = None,
    args = None
) -> Tuple[torch.nn.Module, BreslowEstimator]:
    print('\n \n ***evaluation_visits_from_now***\n \n', evaluation_visits_from_now)
    
    assert testrun is not None, "testrun argument must be provided as True or False. It is currently None. Please set testrun to True for a quick test run or False for a full run."
    assert sweep is not None, "sweep argument must be provided as True or False. It is currently None. Please set testrun to True for a quick test run or False for a full run."

    if testrun:
        epochs = 2 if epochs > 2 else epochs
        print(f"Only running 10 train and eval batches for {epochs} epochs for debugging.")

    best_score = np.inf
    best_model = None
    best_breslow = None
    best_encoder = None
    no_improvement = 0
    print('in train_survival_model considering  evaluation_visits_from_now as', evaluation_visits_from_now)

    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        breslow, survival_curves, train_loss = train_one_epoch(
            survival_model, encoder, optimizer, 
            train_loader, device, 
            logger, epoch, testrun, ssl_method,
            tcn_loss = tcn_loss,
            data_name = data_name, 
            evaluation_visits_from_now = evaluation_visits_from_now,
            args = args
        )
        out_dict, performance = Evaluate(
            survival_model=survival_model,
            encoder=encoder,
            breslow=breslow,
            dataloaders={"val": val_loader},
            device=device,
            logger=logger,
            epoch=epoch,
            train_data=train_data,
            testrun=testrun,
            ssl_method = ssl_method,
            data_name = data_name,
            evaluation_visits_from_now = evaluation_visits_from_now
        )

        val_score = performance["val"]["ibs"]
        brier_scores = performance["val"]["brier_scores"]
        print(f"Epoch {epoch+1} Train loss: {train_loss}, Val ibs: {val_score}, brier scores {brier_scores}")
        logger.log({"val/ibs": val_score}, step=epoch)
        logger.log({"val/mean_auroc": performance["val"]["mean_auroc"]}, step=epoch)
        logger.log({"val/brier_score_mean": brier_scores.mean()}, step=epoch)
        
        logger.log({
                f"val/brier_score_t{evaluation_visits_from_now[i]}": v 
                for i, v in enumerate(brier_scores)
            }, step=epoch)
        logger.log({'train/loss': train_loss}, step=epoch)

        #saving latest results
        chkpt_dir = logger.log_path.parent
        if (not testrun):
            if (not sweep):
                if (train_size is None) or (int(train_size) == 1000):
                    print('saving latest model')
                    checkpoint = {
                        "epoch":epoch,
                        "latest_surv_model": survival_model.state_dict(),
                        "latest_encoder": encoder.state_dict(),
                        "latest_optimizer0": optimizer[0].state_dict(),
                        "latest_optimizer1": optimizer[1].state_dict(),
                        "latest_breslow": breslow,
                    }
                    torch.save(checkpoint, chkpt_dir / "latest_model.pt")

        if val_score < best_score:
            best_score = val_score
            best_epoch = epoch
            best_model = copy.deepcopy(survival_model)
            best_breslow = copy.deepcopy(breslow)
            best_encoder = copy.deepcopy(encoder)
            no_improvement = 0

            # Save best model
            chkpt_dir = logger.log_path.parent
            
            if (not testrun):
                if (not sweep):
                    if (train_size is None) or (int(train_size) == 1000):
                        print('Not saving model checkpoint due to sweep or testrun settings.')
                        checkpoint = {
                        "epoch": best_epoch,
                        "best_surv_model": best_model.state_dict(),
                        "best_encoder": best_encoder.state_dict(),
                        "best_optimizer0": optimizer[0].state_dict(),
                        "best_optimizer1": optimizer[1].state_dict(),
                        "best_breslow": breslow,
                    }
                        torch.save(checkpoint, chkpt_dir / "best_model.pt")

            # Log the best score
            brier_scores_best = performance["val"]["brier_scores"]
            logger.log({"val/ibs_best": performance["val"]["ibs"]}, step=epoch)
            logger.log({"val/mean_auroc_best": performance["val"]["mean_auroc"]}, step=epoch)
            logger.log({
                    f"val/brier_score_t{i}": v 
                    for i, v in enumerate(brier_scores_best)
                }, step=epoch)
            logger.log({"best_epoch": best_epoch}, step=epoch)

        else:
            no_improvement += 1
            
            if no_improvement >= early_stopping_patience:
                print(f"Early stopping after {epoch+1} epochs.")
                break

    return best_encoder, best_model, best_breslow



def train_one_epoch_tcn(
    survival_model: torch.nn.Module,
    extractor_model: torch.nn.Module,
    tcn_model: torch.nn.Module,
    encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    te_optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    temporal_loader: torch.utils.data.DataLoader,
    device: str,
    logger: Logger,
    epoch: int,
    test_run: bool = False,
    model_str_: str = '',
) -> Tuple[BreslowEstimator, np.ndarray, float]:
    loss_logging = []
    _logits = []
    _events = []
    _durations = []

    survival_model.train()
    encoder.train()

    for (imgs, grades, events, durations, paths), (seq_imgs, seq_mask, seq_labels) in (tqdm(zip(train_loader, temporal_loader), desc="Training Batches")
    ):
        _events.extend(events)
        _durations.extend(durations)

        imgs = imgs.to(device)
        events, durations = events.to(device), durations.to(device)

        seq_imgs = seq_imgs.to(device)

        # Get basenames
        paths = [p.split("/")[-1] for p in paths]

        # Get embeddings
        if 'mae' in model_str_:
            print('in train mae')
            features, _ = encoder(imgs, use_mask = False) # pass images to encoder
            embeddings = features[0]#.reshape((64, 192)) # shape is batch_size x output encoder node
        else:
            embeddings = encoder(imgs)

        for opt in optimizer:
            opt.zero_grad()

        te_optimizer.zero_grad()
        log_partial_hazards = survival_model(embeddings) #pass embeddings to MLP
        _logits.extend(log_partial_hazards)

        loss = get_survival_loss(log_partial_hazards, events, durations, device)


        B, T, C, H, W = seq_imgs.shape
        seq_imgs_flat = seq_imgs.view(B*T, C, H, W) # flattened images
        seq_feats = extractor_model(seq_imgs_flat).view(B, -1, T)  # [Batch_size, Extractor dim, Tempral length]
        # print('seq_feat shape', seq_feats.shape)


        optimal_repr, attention_weights = tcn_model(seq_feats, return_attention=True)
        tcn_loss = mmd_loss(optimal_repr, log_partial_hazards)

        t_loss = tcn_loss + loss
        loss_logging.append(t_loss.item())

        t_loss.backward()


        for opt in optimizer:
            opt.step()
        te_optimizer.step()

        # if test_run and i > 10:
        #     break

    # Fit Breslow estimator
    l = torch.tensor(_logits).detach().cpu()
    breslow = fit_breslow(l, torch.tensor(_durations), torch.tensor(_events))
    evaluation_visits_from_now = [2, 4, 6, 8, 10]
    survival_curves = predict_survival(breslow, torch.tensor(_logits), evaluation_visits_from_now)

    avg_loss = np.mean(loss_logging)
    logger.log({"train_loss": avg_loss}, step=epoch)

    return breslow, survival_curves, avg_loss

def train_survival_model_tcn(
    survival_model: torch.nn.Module,
    extractor_model: torch.nn.Module, 
    tcn_model: torch.nn.Module,
    encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    te_optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    temporal_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: str,
    epochs: int,
    early_stopping_patience: int,
    logger: Logger,
    train_data=None,
    test_run: bool = False,
    model_str_: str = 'mae',
    tcn_loss = None
) -> Tuple[torch.nn.Module, BreslowEstimator]:

    if test_run:
        epochs = 2 if epochs > 2 else epochs
        print(f"Only running 10 train and eval batches for {epochs} epochs for debugging.")

    best_score = np.inf
    best_model = None
    best_breslow = None
    no_improvement = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")

        breslow, survival_curves, train_loss = train_one_epoch_tcn(
            survival_model, extractor_model, tcn_model, encoder, optimizer, te_optimizer, train_loader, temporal_loader, device, logger, epoch, test_run, model_str_
        )
        val_survival_curves, performance = Evaluate(
            survival_model=survival_model,
            encoder=encoder,
            breslow=breslow,
            dataloaders={"val": val_loader},
            device=device,
            logger=logger,
            epoch=epoch,
            train_data=train_data,
            test_run=test_run,
            model_str_=model_str_
        )

        val_score = performance["val"]["ibs"]

        print(f"Epoch {epoch+1} Train loss: {train_loss}, Val ibs: {val_score}")

        if val_score < best_score:
            best_score = val_score
            best_model = survival_model
            best_breslow = breslow
            no_improvement = 0

            # Save best model
            chkpt_dir = logger.log_path.parent
            checkpoint = {
                "model": survival_model.state_dict(),
                "extractor_model": extractor_model.state_dict(),
                "tcn_model": tcn_model.state_dict(),
                # "optimizer": optimizer.state_dict(),
                "breslow": breslow,
            }
            torch.save(checkpoint, chkpt_dir / "best_model.pt")

            # Log the best score
            logger.log({"val/ibs_best": performance["val"]["ibs"]}, step=epoch)
            logger.log({"val/mean_auroc_best": performance["val"]["mean_auroc"]}, step=epoch)

        else:
            no_improvement += 1
            if no_improvement >= early_stopping_patience:
                print(f"Early stopping after {epoch+1} epochs.")
                break

    return best_model, best_breslow


