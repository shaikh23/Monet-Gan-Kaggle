# Monet Style GAN - CycleGAN Implementation

A CycleGAN implementation for transforming photographs into Monet-style paintings using unpaired image-to-image translation. This project was developed for the Kaggle "I'm Something of a Painter Myself" competition.

## Project Overview

This notebook implements CycleGAN for artistic style transfer, specifically converting natural landscape photographs into impressionistic Monet-style paintings. The model learns bidirectional mappings between photo and painting domains without requiring paired training examples.

The competition task requires generating 7,000-10,000 Monet-style images at 256×256 resolution, evaluated using MiFID (Memorization-informed Fréchet Inception Distance).

## Dataset

The project uses the Kaggle GAN Getting Started competition dataset:

- **Monet Paintings**: 300 impressionist paintings (256×256 RGB)
- **Photographs**: 7,028 natural landscape photos (256×256 RGB)
- **Format**: Both TFRecord and JPEG available (TFRecord used for training)
- **Pairing**: Unpaired dataset (no corresponding photo-painting pairs)

### Dataset Characteristics

**Monet Domain**: Soft brushstrokes, emphasis on light and color, impressionistic style
**Photo Domain**: Sharp details, realistic colors, diverse natural landscapes

Statistical analysis shows photographs have higher variance (sharper contrast) and different color distributions compared to Monet paintings, which exhibit softer textures and more uniform intensity.

## Model Architecture

### CycleGAN Components

**Generators (2)**:
- ResNet-based encoder-decoder architecture
- 6 residual blocks for 256×256 images
- Instance normalization throughout
- Reflection padding to reduce border artifacts
- Tanh output activation for [-1,1] range
- ~11M parameters per generator

**Discriminators (2)**:
- 70×70 PatchGAN architecture
- Evaluates style quality at patch level
- Instance normalization (skip first layer)
- LeakyReLU activations

### Loss Functions

1. **Adversarial Loss**: LSGAN (MSE-based) for stable training gradients
2. **Cycle Consistency Loss** (λ=10): Ensures F(G(x)) ≈ x and G(F(y)) ≈ y
3. **Identity Loss** (λ=5): Prevents unnecessary color shifts for images already in target domain

Total generator loss combines all three components with specified weights.

### Architecture Selection Rationale

CycleGAN was chosen over alternatives (DCGAN, Pix2Pix, StyleGAN) because:
- Works with unpaired data (dataset has no corresponding photo-painting pairs)
- Cycle consistency preserves content structure while transferring style
- Bidirectional learning provides stronger training signal
- Proven effectiveness on artistic style transfer tasks

## Training Configuration

**Hyperparameters**:
- Learning rate: 2×10⁻⁴ (Adam optimizer, β₁=0.5)
- Batch size: 1 (standard for instance normalization)
- Image size: 256×256×3
- Loss weights: λ_cycle=10.0, λ_identity=5.0
- Epochs: 1 (demo), Steps: 150

**Hardware**: GPU-accelerated training (Tesla P100 in notebook example)

## Results

### Training Metrics (Epoch Averages)

- Discriminator X: 0.514
- Discriminator Y: 0.530
- Generator adversarial: 0.610 (Photo→Monet), 0.544 (Monet→Photo)
- Cycle consistency: 0.697
- Identity loss: 0.673
- Training time: 100.8s (150 steps)

### Loss Analysis

**Discriminator losses near 0.5**: Indicates healthy equilibrium between generator and discriminator
**Generator adversarial losses 0.5-0.6**: Generators successfully fool discriminators
**Cycle consistency 0.697**: Reasonable content preservation with room for improvement

### Visual Quality

The model successfully demonstrates:
- Color transfer from realistic to impressionistic tones
- Content preservation (objects and composition remain recognizable)
- Reduction in high-frequency photographic details
- No checkerboard artifacts or border issues
- Consistent style across diverse inputs

Areas for improvement with extended training:
- Finer brushstroke textures
- Enhanced edge softening
- More vibrant color intensity
- Further reduction of photographic sharpness



## Implementation Details

### Data Loading
- TFRecord format for optimized I/O performance
- Images normalized to [-1,1] range for tanh activation
- Shuffle and repeat for training, no augmentation applied

### Training Loop
- Alternating updates for all four networks
- Persistent gradient tape for simultaneous loss computation
- Progress monitoring every 50 steps
- No learning rate scheduling (constant 2×10⁻⁴)

### Custom Layers
- ReflectionPadding2D: Mirrors edge pixels to reduce border artifacts
- InstanceNormalization: Normalizes per-image statistics (custom fallback if TensorFlow Addons unavailable)
- ResNet blocks: Skip connections for gradient flow in deep generators

## Future Improvements

1. **Extended training**: 5,000-10,000 steps for refined details
2. **Data augmentation**: Horizontal flip, random crop, color jitter
3. **Learning rate scheduling**: Linear decay after 50% training
4. **Hyperparameter tuning**: Test different λ_cycle and λ_identity values
5. **Perceptual loss**: Add VGG-based loss for improved texture realism
