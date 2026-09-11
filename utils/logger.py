import yaml
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
import os
warnings.filterwarnings(
    "ignore",
    message="Setting an item of incompatible dtype is deprecated",
    category=FutureWarning,
)


def flatten_dict(d, parent_key='', sep='_'):
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            # Recursively flatten nested dicts
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, np.ndarray):
            # Convert arrays to lists for JSON serialization
            items.append((new_key, v.tolist()))
        elif isinstance(v, (np.float64, np.int64)):
            # Convert numpy types to Python types
            items.append((new_key, float(v)))
        else:
            items.append((new_key, v))
    return dict(items)

class Logger(object):
    def init(
        self,
        project_name: str = "survival-on-embeddings-",
        timestamp: str = None,
        config: dict = None,
        use_wandb: bool = True,
        tags: list = None,
        run_name: str = None,
        run_id: str = None,
        chkpt_path: str = None,
        log_local: str = None

    ):
        self.use_wandb = use_wandb
        self.config = config
        self.log_local = log_local
        self.timestamp = (
            timestamp if timestamp is not None else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        self.project_name = project_name
        self.run_name = run_name
        self.name_ = self.run_name + "_" + self.timestamp

        if run_id is None:
            self.log_path = Path(f"checkpoints/{self.project_name}/{self.name_}/log.csv")
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            print('\n \n ***log path parent*** \n \n', self.log_path.parent)

            with open(self.log_path.parent / "config.yaml", "w") as file:
                yaml.dump(config, file)
           
            if self.use_wandb:
                try:
                    import wandb
                except ImportError:
                    self.use_wandb = False
                    print("wandb not installed. Falling back to logging to CSV.")


            if self.use_wandb:
                import wandb
                run_id = config.get("run_id", wandb.util.generate_id())
                config["run_id"] = run_id
                if wandb.run is None: #checks whether a wandb run has already been started
                    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                        wandb.init(project=self.project_name, config=config, name= self.name_, tags=tags, save_code=True, id=run_id, resume="allow",)
                else:
                    # Update the name of the run and add the new config values
                    wandb.run.name = self.timestamp
                    for key, value in config.items():
                        if key not in wandb.config.keys():
                            wandb.config[key] = value
                    # Add tags
                    if tags is not None:
                        wandb.run.tags = tags
        else:
            print(f"resuming wandb from id {run_id}")
            self.log_path = Path(f"{chkpt_path}/new_logs.csv")
            with open(self.log_path.parent / "config_new.yaml", "w") as file:
                yaml.dump(config, file)
            if self.use_wandb:
                try:
                    import wandb
                except ImportError:
                    self.use_wandb = False
                    print("wandb not installed. Falling back to logging to CSV.")


            if self.use_wandb:
                import wandb
                run_id = config.get("run_id", wandb.util.generate_id())
                config["run_id"] = run_id
                if wandb.run is None: #checks whether a wandb run has already been started
                    wandb.init(project=self.project_name, config=config, name= self.name_, tags=tags, save_code=True, id=run_id, resume="must",)
                else:
                    # Update the name of the run and add the new config values
                    wandb.run.name = self.timestamp
                    for key, value in config.items():
                        if key not in wandb.config.keys():
                            wandb.config[key] = value
                    # Add tags
                    if tags is not None:
                        wandb.run.tags = tags
            
        print(f"Logger initialized. Logging to: {self.log_path}")

    def log(self, metrics: dict, split: str = None, step: int = None):
        # print('logging metrics:', metrics)
        if split is not None:
            metrics = {f"{split}/{key}": value for key, value in metrics.items()}
        
        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=step)
        else:
            # Load config and flatten it
            config_path = self.log_path.parent / "config.yaml"
            if config_path.exists():
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                # Prefix config keys to distinguish from metrics
                # print('config', config)
                config_flat = {f"{k}": v for k, v in config.items()}
            else:
                config_flat = {}
            
            # Combine config and metrics
            combined = {**config_flat, **metrics}
            
            # Create log file if it doesn't exist
            if not self.log_path.exists():
                pd.DataFrame(columns=combined.keys()).to_csv(self.log_path, index=False)
            
    
            df = pd.read_csv(self.log_path)
            
            # Determine row index
            row_index = len(df) if step is None else step
            
            # Add combined data to dataframe
            for key, value in combined.items():
                if "aurocs" in key or "scores" in key:
                    if key not in df.columns:
                        df[key] = None
                        df[key] = df[key].astype(object)
                df.at[row_index, key] = value
            
            # Save back to CSV
            df.to_csv(self.log_path, index=False)   

    # def log(self, metrics: dict, split: str = None, step: int = None):
        
    #     if split is not None:
    #         metrics = {f"{split}/{key}": value for key, value in metrics.items()}
    #     if self.use_wandb:
    #         import wandb
    #         wandb.log(metrics, step=step)
    #     else:
    #         if not self.log_path.exists():
    #             pd.DataFrame(columns=metrics.keys()).to_csv(self.log_path, index=False)
    #         df = pd.read_csv(self.log_path)
    #         df = pd.DataFrame(list(self.log_path.parent / "config.yaml".items()), columns=['parameter', 'value'])


    #         row_index = len(df) + 1 if step is None else step
    #         for key, value in metrics.items():
    #             if "aurocs" in key or "scores" in key:
    #                 df[key] = None
    #                 df[key] = df[key].astype(object)
    #             df.at[row_index, key] = value
    #         df.to_csv(self.log_path, index=False)

    def log_final_results(self, final_metrics: dict, file_path: str = "reports.csv"):
        """
        Log final results directly to CSV
        """
        
        flat_metrics = flatten_dict(final_metrics)
        # print('*** \n \n flat_metrics \n \n ***', flat_metrics)
        result = {
            'run_name': self.name_,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        if self.config is not None:
            for key, value in self.config.items():
                result[f'{key}'] = value
        
        result.update(flat_metrics)
        
        df_new = pd.DataFrame([result])
        
        if not Path(f'{file_path}').exists():
            df_new.to_csv(f'{file_path}', index=False)
        else:
            df_new.to_csv(f'{file_path}', mode='a', header=False, index=False)
        
        print(f"Final results saved to {file_path}")

    def log_survival_curves(self, curves_df: pd.DataFrame, split: str):
        curves_df.to_csv(self.log_path.parent / f"{split}_survival_curves.csv", index=False)
    def log_survival_curves_test(self, curves_df: pd.DataFrame, split: str):
        curves_df.to_csv(self.log_local / f"{split}_survival_curves.csv", index=False)