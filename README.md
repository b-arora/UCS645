# Assignment 2 – Learning Probability Density Functions using GAN

## Objective
The objective of this assignment is to learn an unknown probability density function (PDF) of a transformed random variable using only data samples, without assuming any parametric distribution. A Generative Adversarial Network (GAN) is used to implicitly model the distribution.

## Dataset
- Dataset: India Air Quality Data
- Feature used: NO₂ concentration
- File used: data.csv

## Data Transformation
Each NO₂ value x is transformed into z using the function:

z = x + a_r sin(b_r x)

For roll number 102303847:
- a_r = 0.5
- b_r = 0.9

Final transformation:
z = x + 0.5 sin(0.9x)

The NO₂ data is normalized to zero mean and unit variance before applying the transformation.

## Methodology

### GAN Architecture
Generator:
- Input: 1D Gaussian noise ~ N(0,1)
- Two hidden layers with 64 neurons each and ReLU activation
- Output: Generated sample z_f

Discriminator:
- Input: Real or generated z samples
- Two hidden layers with 64 neurons each and ReLU activation
- Output: Probability of sample being real (Sigmoid activation)

### Training Details
- Loss Function: Binary Cross Entropy
- Optimizer: Adam
- Learning Rate: 0.0002
- Epochs: 3000
- Batch Size: 128

The discriminator learns to distinguish real transformed samples from generated samples, while the generator learns to fool the discriminator.

## PDF Approximation
After training:
- 10,000 samples are generated using the trained generator
- Kernel Density Estimation (KDE) is applied to estimate the probability density function
- The estimated PDF is plotted over the range of generated samples

## Files
- gan_assn2.ipynb – Implementation notebook
- data.csv – Dataset
- README.md – Project description

## Observations
- The GAN is able to learn the underlying shape of the transformed data distribution
- Mode coverage is reasonable for a simple network architecture
- Training remains stable due to normalization and balanced learning rates
- Generated samples closely resemble the real transformed samples

## Notes
- The probability density is learned only from samples
- No parametric distribution is assumed
- The implementation follows the assignment guidelines
