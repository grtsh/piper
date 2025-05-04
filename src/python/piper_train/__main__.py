import argparse
import json
import logging
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from .vits.lightning import VitsModel

_LOGGER = logging.getLogger(__package__)


def main():
    logging.basicConfig(level=logging.DEBUG)
    _LOGGER.info("Starting Piper training script...") # Added info message

    parser = argparse.ArgumentParser(prog="piper_train") # Added prog name
    parser.add_argument(
        "--dataset-dir", required=True, help="Path to pre-processed dataset directory"
    )
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        default=None, # Default to None if not provided
        help="Save checkpoint every N epochs (default: None)",
    )
    parser.add_argument(
        "--quality",
        default="medium",
        choices=("x-low", "medium", "high"),
        help="Quality/size of model (default: medium)",
    )
    parser.add_argument(
        "--resume_from_single_speaker_checkpoint",
        help="For multi-speaker models only. Converts a single-speaker checkpoint to multi-speaker and resumes training",
    )
    # Add PyTorch Lightning Trainer arguments
    # Deprecated in newer Lightning versions, but keeping if required by old piper version
    # Consider using Trainer(accelerator="gpu", devices=...) directly if updating Lightning
    try:
       Trainer.add_argparse_args(parser)
    except AttributeError:
        _LOGGER.warning("Trainer.add_argparse_args is deprecated. Consider updating Lightning and passing Trainer args directly.")
        # Add common args manually if needed for older compatibility or clarity
        parser.add_argument('--gpus', type=int, default=None, help='Number of gpus')
        parser.add_argument('--max_epochs', type=int, default=None, help='Max training epochs')
        parser.add_argument('--default_root_dir', type=str, default=None, help='Default root directory for logs and checkpoints')
        parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint to resume from')


    # Add VitsModel specific arguments (make sure VitsModel has the add_model_specific_args method)
    if hasattr(VitsModel, "add_model_specific_args"):
        VitsModel.add_model_specific_args(parser)
    else:
        _LOGGER.warning("VitsModel does not have add_model_specific_args method. Model-specific args might be missing.")
        # Add essential ones manually if needed
        parser.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU") # Example default
        parser.add_argument("--num-workers", type=int, default=4, help="Num workers for dataloader") # Example default


    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    _LOGGER.debug("Parsed arguments: %s", args)

    args.dataset_dir = Path(args.dataset_dir)
    # Set default_root_dir for logs/checkpoints if not provided
    # Saving inside the dataset dir might be convenient
    if args.default_root_dir is None:
        args.default_root_dir = args.dataset_dir / "lightning_logs" # Save logs inside dataset dir
        _LOGGER.info("Default root directory not set, using: %s", args.default_root_dir)
    args.default_root_dir = Path(args.default_root_dir) # Ensure it's a Path object

    # Ensure the root dir exists
    args.default_root_dir.mkdir(parents=True, exist_ok=True)

    # --- Reproducibility ---
    torch.manual_seed(args.seed)
    # Ensure cuda seeds are set if using GPU - might be handled by Trainer's seed everything
    # pl.seed_everything(args.seed, workers=True) # Newer Lightning way

    # Benchmark mode can potentially speed up training if input sizes don't vary much
    torch.backends.cudnn.benchmark = True

    # --- Load Config ---
    config_path = args.dataset_dir / "config.json"
    dataset_path = args.dataset_dir / "dataset.jsonl" # Assuming preprocessed dataset index

    if not config_path.is_file():
        _LOGGER.error("Config file not found at %s", config_path)
        return
    if not dataset_path.is_file():
        _LOGGER.error("Dataset index file not found at %s", dataset_path)
        return

    _LOGGER.info("Loading config from %s", config_path)
    with open(config_path, "r", encoding="utf-8") as config_file:
        try:
            config = json.load(config_file)
            num_symbols = int(config["num_symbols"])
            num_speakers = int(config["num_speakers"])
            sample_rate = int(config["audio"]["sample_rate"])
        except (KeyError, json.JSONDecodeError, TypeError) as e:
             _LOGGER.error("Failed to parse config file %s: %s", config_path, e)
             return

    _LOGGER.info("Config loaded: Symbols=%d, Speakers=%d, SampleRate=%d",
                 num_symbols, num_speakers, sample_rate)


    # --- Configure Callbacks ---
    callbacks = []

    # 1. Save the best model based on validation loss
    best_checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # Metric to monitor
        mode="min",          # Lower val_loss is better
        save_top_k=1,        # Save only the single best model checkpoint
        filename="best_model-{epoch:02d}-{val_loss:.2f}", # Filename for the best model
        verbose=True,        # Print message when a checkpoint is saved as 'best'
        dirpath=args.default_root_dir / "checkpoints" # Explicitly set checkpoint dir
    )
    callbacks.append(best_checkpoint_callback)
    _LOGGER.info("Configured to save the best model based on 'val_loss'.")

    # 2. Always save the last checkpoint (useful for resuming)
    last_checkpoint_callback = ModelCheckpoint(
        filename="last-{epoch:02d}-{step}",
        save_last=True, # Creates/updates 'last.ckpt' symbolic link
        dirpath=args.default_root_dir / "checkpoints" # Explicitly set checkpoint dir
    )
    callbacks.append(last_checkpoint_callback)
    _LOGGER.info("Configured to save the last model checkpoint (last.ckpt).")

    # 3. Optionally save checkpoints periodically
    if args.checkpoint_epochs is not None and args.checkpoint_epochs > 0:
        periodic_checkpoint_callback = ModelCheckpoint(
            every_n_epochs=args.checkpoint_epochs,
            filename="periodic-{epoch:02d}-{step}", # Filename for periodic saves
            save_top_k=-1, # Ensure it saves regardless of metric, -1 saves all that match the rule
            dirpath=args.default_root_dir / "checkpoints" # Explicitly set checkpoint dir
        )
        callbacks.append(periodic_checkpoint_callback)
        _LOGGER.info(
            "Periodic checkpoints will be saved every %s epoch(s).", args.checkpoint_epochs
        )
    else:
         _LOGGER.info("Periodic epoch checkpointing is disabled (--checkpoint-epochs not set or <= 0).")


    # --- Instantiate Trainer ---
    # Pass callbacks list here
    # Handle potential deprecation of from_argparse_args
    trainer_kwargs = {
        "default_root_dir": str(args.default_root_dir),
        "callbacks": callbacks,
        # Map common args manually if add_argparse_args was skipped or deprecated
        "max_epochs": getattr(args, 'max_epochs', -1), # Default to run indefinitely if not set
        "accelerator": "gpu" if getattr(args, 'gpus', 0) > 0 else "cpu",
        "devices": getattr(args, 'gpus', 1) if getattr(args, 'gpus', 0) > 0 else None,
        # Add other Trainer args as needed, e.g., precision, strategy...
        # "precision": 16 if getattr(args, 'precision', 32) == 16 else 32,
    }
    _LOGGER.info("Instantiating Trainer with args: %s", trainer_kwargs)
    try:
        # Try using from_argparse_args if it exists and no conflicts
        # Note: Passing explicit args like callbacks overrides those from args
        trainer = Trainer.from_argparse_args(args, **trainer_kwargs)
        _LOGGER.info("Trainer instantiated using from_argparse_args.")
    except (AttributeError, TypeError):
         _LOGGER.warning("Trainer.from_argparse_args failed or is unavailable. Instantiating Trainer directly.")
         # Remove args that might conflict if passed directly and via from_argparse_args
         trainer_kwargs.pop('gpus', None) # devices handles this
         trainer = Trainer(**trainer_kwargs)
         _LOGGER.info("Trainer instantiated directly.")


    # --- Prepare Model Arguments ---
    # Convert Namespace to dict, filtering out None values potentially
    dict_args = {k: v for k, v in vars(args).items() if v is not None}

    # Apply quality presets by overriding specific model args
    if args.quality == "x-low":
        _LOGGER.info("Applying 'x-low' quality preset.")
        dict_args["hidden_channels"] = 96
        dict_args["inter_channels"] = 96
        dict_args["filter_channels"] = 384
    elif args.quality == "high":
        _LOGGER.info("Applying 'high' quality preset.")
        dict_args["resblock"] = "1"
        dict_args["resblock_kernel_sizes"] = (3, 7, 11)
        dict_args["resblock_dilation_sizes"] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        )
        dict_args["upsample_rates"] = (8, 8, 2, 2)
        dict_args["upsample_initial_channel"] = 512
        dict_args["upsample_kernel_sizes"] = (16, 16, 4, 4)
    else: # Medium quality (default)
         _LOGGER.info("Using 'medium' quality preset (default model parameters).")


    # --- Instantiate Model ---
    _LOGGER.info("Instantiating VitsModel...")
    try:
        model = VitsModel(
            num_symbols=num_symbols,
            num_speakers=num_speakers,
            sample_rate=sample_rate,
            dataset=[str(dataset_path)], # Pass dataset path as string list
            **dict_args, # Pass all relevant args from command line and presets
        )
        _LOGGER.info("VitsModel instantiated successfully.")
    except Exception as e:
        _LOGGER.exception("Failed to instantiate VitsModel: %s", e)
        return


    # --- Handle Resuming from Single-Speaker Checkpoint ---
    # This logic seems specific and complex, keeping it as is.
    # Ensure the 'load_state_dict' helper function is defined below main or imported.
    if args.resume_from_single_speaker_checkpoint:
        _LOGGER.info("Handling resume from single-speaker checkpoint logic...")
        if not Path(args.resume_from_single_speaker_checkpoint).is_file():
             _LOGGER.error("Single-speaker checkpoint not found: %s", args.resume_from_single_speaker_checkpoint)
             return

        assert (
            num_speakers > 1
        ), "--resume_from_single_speaker_checkpoint is only for multi-speaker models. Use --resume_from_checkpoint for single-speaker models."

        # Load single-speaker checkpoint
        _LOGGER.debug(
            "Loading single-speaker checkpoint state: %s",
            args.resume_from_single_speaker_checkpoint,
        )
        try:
            # Load checkpoint without instantiating the model fully if possible
            # Or load into a temporary model instance
            model_single = VitsModel.load_from_checkpoint(
                args.resume_from_single_speaker_checkpoint,
                map_location="cpu", # Load to CPU first
                dataset=None, # Don't need dataset for state dict
                # Provide necessary dummy args if load_from_checkpoint requires them
                num_symbols=num_symbols,
                num_speakers=1, # Important: Load as if it were single speaker
                sample_rate=sample_rate,
            )

            g_dict = model_single.model_g.state_dict()
            d_dict = model_single.model_d.state_dict()

            # Filter generator state dict
            filtered_g_dict = {}
            for key, value in g_dict.items():
                if not (
                    key.startswith("dec.cond")
                    or key.startswith("dp.cond")
                    or ("enc.cond_layer" in key)
                    or ("emb_g" in key) # Filter speaker embedding too
                ):
                    filtered_g_dict[key] = value
                else:
                    _LOGGER.debug("Excluding key from single-speaker G: %s", key)


            # Load filtered state dicts into the multi-speaker model
            # Use strict=False to ignore missing keys (like speaker embeddings in G)
            # and unexpected keys (potentially new layers in multi-speaker).
            _LOGGER.info("Loading Generator state dict (excluding speaker-specific keys)...")
            model.model_g.load_state_dict(filtered_g_dict, strict=False)

            _LOGGER.info("Loading Discriminator state dict...")
            model.model_d.load_state_dict(d_dict, strict=True) # D should usually match

            _LOGGER.info(
                "Successfully loaded states from single-speaker checkpoint into multi-speaker model."
            )
            # Explicitly set the checkpoint path for the Trainer to potentially load optimizer states etc.
            # Or maybe not, as we only transferred weights. Let Trainer start fresh optimizers.
            args.resume_from_checkpoint = None # Don't use Lightning's resume for this case
            _LOGGER.warning("Resuming from single-speaker checkpoint loads weights only. Optimizer state is NOT restored.")

        except Exception as e:
            _LOGGER.exception("Error during single-speaker checkpoint loading: %s", e)
            return


    # --- Start Training ---
    _LOGGER.info("Starting trainer.fit()...")
    try:
        # Determine checkpoint path for standard resuming (not single->multi speaker case)
        resume_ckpt_path = getattr(args, 'resume_from_checkpoint', None)
        if resume_ckpt_path == "last":
            resume_ckpt_path = args.default_root_dir / "checkpoints" / "last.ckpt"
            if not resume_ckpt_path.is_file():
                _LOGGER.warning("Attempted to resume from 'last', but last.ckpt not found. Starting fresh.")
                resume_ckpt_path = None
            else:
                 _LOGGER.info("Resuming from last checkpoint: %s", resume_ckpt_path)
        elif resume_ckpt_path and not Path(resume_ckpt_path).is_file():
             _LOGGER.warning("Resume checkpoint path specified but not found: %s. Starting fresh.", resume_ckpt_path)
             resume_ckpt_path = None
        elif resume_ckpt_path:
             _LOGGER.info("Resuming from specified checkpoint: %s", resume_ckpt_path)


        trainer.fit(model, ckpt_path=str(resume_ckpt_path) if resume_ckpt_path else None)
        _LOGGER.info("Training finished.")
    except Exception as e:
         _LOGGER.exception("Exception during training: %s", e)


# Helper function for loading state dict (needed for single-speaker resume)
# Keep this function defined, maybe move it outside main if preferred.
def load_state_dict(model, saved_state_dict):
    """Loads state dict selectively, keeping existing keys if not found in saved dict."""
    state_dict = model.state_dict()
    new_state_dict = {}
    loaded_keys = set()

    for k, v in saved_state_dict.items():
        if k in state_dict:
            if state_dict[k].shape == v.shape:
                new_state_dict[k] = v
                loaded_keys.add(k)
            else:
                _LOGGER.warning(f"Shape mismatch for key {k}: Model has {state_dict[k].shape}, Checkpoint has {v.shape}. Skipping.")
        else:
             _LOGGER.warning(f"Key {k} from checkpoint not found in the current model. Skipping.")


    missing_keys = set(state_dict.keys()) - loaded_keys
    if missing_keys:
         _LOGGER.warning(f"Missing keys in checkpoint, using initialized values for: {missing_keys}")
         for k in missing_keys:
             new_state_dict[k] = state_dict[k] # Use initialized value

    model.load_state_dict(new_state_dict)


def load_state_dict(model, saved_state_dict):
    state_dict = model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        if k in saved_state_dict:
            # Use saved value
            new_state_dict[k] = saved_state_dict[k]
        else:
            # Use initialized value
            _LOGGER.debug("%s is not in the checkpoint", k)
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
