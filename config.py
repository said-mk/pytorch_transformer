from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 160,
        "d_model": 512,
        "datasource": "HausaNLP/HausaVG",
        "lang_src": "en_text",
        "lang_tgt": "ha_text",
        "vocab_size_src": 30000,
        "vocab_size_tgt": 30000,
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_folder = model_folder.replace('/', '_')
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_folder = model_folder.replace('/', '_')
    model_folder_path = Path('.') / model_folder
    
    if not model_folder_path.exists():
        return None
    
    weights_files = list(model_folder_path.glob(f"{config['model_basename']}*.pt"))
    if len(weights_files) == 0:
        return None
    
    weights_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
    return str(weights_files[-1])
