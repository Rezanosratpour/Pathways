# UnetConvLSTM Quick Start Guide

This guide shows how to use the simplified high-level API for UnetConvLSTM models.

## Installation

The high-level API is in `UNET_torch_api.py`. Make sure all dependencies are installed:

```bash
pip install torch pytorch h5py numpy
```

## Quick Examples

### Example 1: Train with Preset Configuration (Simplest)

```python
from UNET_torch_api import quick_train

# Train with 'fast' preset (quick training)
metrics = quick_train(
    dataset_path='path/to/data.hdf5',
    preset='fast',
    model_name='my_model'
)

print(f"MSE: {metrics['MSE']}")
print(f"R²: {metrics['R2']}")
```

**Available presets:**
- `small` - Small model for testing
- `medium` - Balanced configuration (default)
- `large` - Large model for accuracy
- `fast` - Quick training (few epochs)
- `accurate` - Maximum accuracy (many epochs)
- `full_ram` - Uses all system RAM (fastest but requires lots of memory)

### Example 2: Advanced Training with Configuration

```python
from UNET_torch_api import ModelConfig, ModelTrainer

# Get a preset and customize
config = ModelConfig.get_preset('medium')
config.dataset_path = 'path/to/data.hdf5'
config.epochs = 25
config.batch_size = 64
config.base_channels = 48

# Create trainer and train
trainer = ModelTrainer(config)
metrics = trainer.train()

# Print training summary
trainer.print_summary()
```

### Example 3: Train with Auxiliary Data

```python
from UNET_torch_api import ModelConfig, ModelTrainer

config = ModelConfig.get_preset('medium')
config.dataset_path = 'path/to/precipitation.hdf5'
config.aux_dataset_path = 'path/to/temperature.hdf5'
config.C = 1  # precipitation channels
config.model_name = 'precip_with_temp'

trainer = ModelTrainer(config)
metrics = trainer.train()
```

### Example 4: Fine-tune Pre-trained Model

```python
from UNET_torch_api import ModelTrainer, ModelConfig

# Load existing config
config = ModelConfig.get_preset('fast')
config.dataset_path = 'path/to/new_data.hdf5'

# Create trainer
trainer = ModelTrainer(config)

# Train from scratch (or load existing model_path)
trainer.model_path = 'path/to/pretrained_model.pt'

# Fine-tune with lower learning rate
metrics = trainer.finetune(
    epochs=10,
    learning_rate=1e-4,
    freeze_encoder=False  # Set True to keep ConvLSTM weights fixed
)
```

### Example 5: Evaluate Pre-trained Model

```python
from UNET_torch_api import quick_evaluate

metrics = quick_evaluate(
    model_path='path/to/model.pt',
    dataset_path='path/to/test_data.hdf5'
)

for metric_name, value in metrics.items():
    print(f"{metric_name}: {value}")
```

### Example 6: Custom Configuration

```python
from UNET_torch_api import ModelConfig, ModelTrainer

# Create custom configuration from scratch
config = ModelConfig(
    dataset_path='path/to/data.hdf5',
    data_type='real',
    seq_len=3,
    H=90,
    W=340,
    C=1,
    base_channels=64,
    batch_size=32,
    epochs=30,
    learning_rate=1e-3,
    mode='ram_safe',  # or 'full_ram'
    num_workers=4,
    model_name='custom_model'
)

# Validate and print summary
config.print_summary()

# Train
trainer = ModelTrainer(config)
metrics = trainer.train()
```

## Configuration Parameters

### Dataset Parameters
- `dataset_path` (str): Path to HDF5 dataset **[required]**
- `aux_dataset_path` (str): Path to auxiliary HDF5 dataset (optional)
- `data_type` (str): Data type - 'real', 'pazzel', etc.

### Model Architecture
- `seq_len` (int): Sequence length / time steps (default: 3)
- `H` (int): Height of spatial domain (default: 90)
- `W` (int): Width of spatial domain (default: 340)
- `C` (int): Number of input channels (default: 1)
- `base_channels` (int): Base channels for U-Net (default: 32)
- `dropout` (float): Dropout rate (default: 0.1)

### Training Parameters
- `batch_size` (int): Batch size (default: 8)
- `epochs` (int): Number of training epochs (default: 20)
- `learning_rate` (float): Learning rate (default: 1e-3)
- `early_stopping_patience` (int): Patience for early stopping (default: 5)

### System Parameters
- `model_dir` (str): Directory to save models (default: './Models')
- `mode` (str): 'ram_safe' or 'full_ram' (default: 'ram_safe')
- `num_workers` (int): Data loading workers (default: 4)
- `device` (str): 'cuda' or 'cpu' (auto-detected)

## Working Examples for Your Project

### Train Cluster 2 Model (No Auxiliary)

```python
from UNET_torch_api import ModelTrainer, ModelConfig

config = ModelConfig.get_preset('large')
config.dataset_path = '/home/ec2-user/Phase_3-9/Phase_3/project3/Datasets/subset_data_cluster_2.hdf5'
config.model_name = 'cluster_2_no_aux'
config.epochs = 25
config.base_channels = 64

trainer = ModelTrainer(config)
metrics = trainer.train()
```

### Train Cluster 2 Model (With Auxiliary Temperature)

```python
from UNET_torch_api import ModelTrainer, ModelConfig

config = ModelConfig.get_preset('large')
config.dataset_path = '/home/ec2-user/Phase_3-9/Phase_3/project3/Datasets/subset_data_cluster_2.hdf5'
config.aux_dataset_path = '/home/ec2-user/Phase_3-9/Phase_3/ERA5_tcwv_cluster_2.h5'
config.model_name = 'cluster_2_with_aux'
config.epochs = 25
config.base_channels = 64

trainer = ModelTrainer(config)
metrics = trainer.train()
```

### Evaluate Model

```python
from UNET_torch_api import quick_evaluate

metrics = quick_evaluate(
    model_path='/home/ec2-user/Phase_3-9/Phase_3/project3/Models/cluster_2_no_aux.pt',
    dataset_path='/home/ec2-user/Phase_3-9/Phase_3/project3/Datasets/subset_data_cluster_2.hdf5'
)

print(f"MSE: {metrics['MSE']:.6f}")
print(f"R²: {metrics['R2']:.6f}")
```

## Common Workflows

### Workflow 1: Quick Prototype

```python
from UNET_torch_api import quick_train, quick_evaluate

# Train quickly
metrics = quick_train('data.hdf5', preset='fast')

# Evaluate
eval_metrics = quick_evaluate('Models/QuickModel.pt', 'data.hdf5')
```

### Workflow 2: Production Model

```python
from UNET_torch_api import ModelTrainer, ModelConfig

# Get best preset
config = ModelConfig.get_preset('accurate')
config.dataset_path = 'data.hdf5'
config.model_name = 'production_model'

# Train
trainer = ModelTrainer(config)
metrics = trainer.train()

# Fine-tune
trainer.finetune(epochs=10, learning_rate=1e-4)

# Evaluate final model
final_metrics = trainer.evaluate()
```

### Workflow 3: Hyperparameter Tuning

```python
from UNET_torch_api import ModelTrainer, ModelConfig

results = {}

for batch_size in [16, 32, 64]:
    config = ModelConfig.get_preset('medium')
    config.dataset_path = 'data.hdf5'
    config.batch_size = batch_size
    config.model_name = f'model_bs{batch_size}'
    
    trainer = ModelTrainer(config)
    metrics = trainer.train()
    results[batch_size] = metrics['MSE']

# Find best batch size
best_bs = min(results, key=results.get)
print(f"Best batch size: {best_bs}")
```

## Utility Functions

### List Available Presets

```python
from UNET_torch_api import list_presets

list_presets()
```

### Check Device Info

```python
from UNET_torch_api import get_device_info

get_device_info()
```

### Convert Config to Dictionary

```python
config = ModelConfig.get_preset('medium')
config_dict = config.to_dict()
print(config_dict)
```

## Tips and Tricks

### Memory Management

If running out of memory:
1. Use `ram_safe` mode instead of `full_ram`
2. Reduce `batch_size`
3. Reduce `base_channels`
4. Reduce `num_workers`

### Speed Up Training

If training is slow:
1. Use `full_ram` mode (if you have enough memory)
2. Increase `batch_size`
3. Reduce `num_workers` (paradoxically can speed up small models)
4. Use `preset='fast'` for fewer epochs

### Improve Accuracy

To get better model accuracy:
1. Use `preset='accurate'` or `preset='large'`
2. Increase `epochs`
3. Decrease `learning_rate` gradually
4. Add auxiliary data
5. Try larger `base_channels`

### Debug Issues

To print detailed configuration:
```python
config = ModelConfig.get_preset('medium')
config.print_summary()
```

To validate configuration before training:
```python
config = ModelConfig(...)
config.validate()  # Raises errors if invalid
```

## API Reference

### `ModelConfig`

Main configuration class.

**Methods:**
- `get_preset(name)` - Get pre-configured setup
- `validate()` - Validate configuration
- `to_dict()` - Convert to dictionary
- `print_summary()` - Print configuration

### `ModelTrainer`

Main trainer class.

**Methods:**
- `train()` - Train model
- `finetune(epochs, learning_rate, freeze_encoder, freeze_backbone)` - Fine-tune model
- `evaluate(model_path)` - Evaluate model
- `print_summary()` - Print training summary

### `quick_train(dataset_path, preset, aux_dataset_path, model_name)`

Simple one-liner training function.

### `quick_evaluate(model_path, dataset_path, aux_dataset_path)`

Simple one-liner evaluation function.

### `list_presets()`

Print available presets.

### `get_device_info()`

Print device information.

## Troubleshooting

### "Dataset not found"
- Check `dataset_path` exists
- Use absolute paths

### "CUDA out of memory"
- Reduce `batch_size`
- Use `ram_safe` instead of `full_ram`
- Use `preset='small'` for fewer channels

### "No trained model found"
- Call `trainer.train()` before `trainer.finetune()`
- Provide `model_path` parameter

### "Auxiliary dataset not found"
- Check `aux_dataset_path` exists
- Check file format is HDF5

## Performance Notes

Typical training times on NVIDIA GPU:
- `fast` preset: 5-10 minutes
- `medium` preset: 15-30 minutes
- `large` preset: 45-90 minutes
- `accurate` preset: 2-4 hours

Training times on CPU will be 5-10x slower.

## Next Steps

1. Start with `quick_train()` for prototyping
2. Use `ModelConfig.get_preset()` for better control
3. Customize `ModelConfig` for fine-tuning
4. Use `ModelTrainer.finetune()` to adapt models
