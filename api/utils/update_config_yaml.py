import yaml
import os
from espnet2.utils.yaml_no_alias_safe_dump import yaml_no_alias_safe_dump


def update_config(config_file: str, model_dir: str) -> None:
    """Update paths in a YAML configuration file to point to new model directory.

    This function reads a YAML configuration file, updates specific file paths to 
    point to new locations within the specified model directory, and writes the 
    updated configuration back to the file.

    The following paths are updated if they exist in the config:
    1. BPE model path
    2. Feature normalization stats file path
    3. Encoder pre-trained model checkpoint path
    4. Decoder pre-trained model checkpoint path

    Args:
        config_file: Path to the YAML configuration file to be updated
        model_dir: Path to the model directory where files will be relocated

    Returns:
        None
    """
    # Read the original configuration file
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

        # Update BPE model path if it exists in config
        _update_path_in_config(
            config,
            key_path=['bpemodel'],
            new_path=os.path.join(model_dir, "bpe_model"),
            description="BPE model"
        )

        # Update feature normalization stats file path if using global_mvn
        if config.get("normalize", None) == "global_mvn":
            _update_path_in_config(
                config,
                key_path=['normalize_conf', 'stats_file'],
                new_path=os.path.join(model_dir, "feat_normalize"),
                description="Feature normalization stats file"
            )

        # Update encoder pre-trained model path if it exists
        _update_path_in_config(
            config,
            key_path=['encoder_conf', 'speech2c_dir_ckpt'],
            new_path=os.path.join(model_dir, "en_pre_trained"),
            description="Encoder pre-trained model"
        )

        # Update decoder pre-trained model path if it exists
        _update_path_in_config(
            config,
            key_path=['decoder_conf', 'speech2c_dir_ckpt'],
            new_path=os.path.join(model_dir, "de_pre_trained"),
            description="Decoder pre-trained model"
        )

    # Write the updated configuration back to file
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(yaml_no_alias_safe_dump(config, indent=4, sort_keys=False))


def _update_path_in_config(
    config: dict,
    key_path: list,
    new_path: str,
    description: str
) -> None:
    """Helper function to update a path in nested config dictionary.

    Args:
        config: The configuration dictionary to modify
        key_path: List of keys representing the path to the value to update
        new_path: The new path to set
        description: Description of the path being updated (for logging)

    Returns:
        None
    """
    current_value = config
    try:
        # Traverse the nested dictionary to find the target key
        for key in key_path[:-1]:
            current_value = current_value[key]

        # Check if the target key exists and has a value
        if key_path[-1] in current_value and current_value[key_path[-1]] is not None:
            old_path = current_value[key_path[-1]]
            print(f"Updating {description} path: {old_path} -> {new_path}")
            current_value[key_path[-1]] = new_path
    except KeyError:
        # Key path doesn't exist in config - skip silently
        pass
