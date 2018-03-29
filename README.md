# colamatch

## Dependencies
* numpy
* scikit-learn
* scikit-image

## Installation

### Installation on Linux / Mac
Clone the repository from github and install with pip:

```bash
git clone https://github.com/FZJ-INM1-BDA/colamatch.git
cd colamatch
pip install -r requirements.txt
pip install .
```

## Development
Clone colamatch from github and install with option -e, so that no reinstallation is needed for every update.
```bash
git clone https://github.com/FZJ-INM1-BDA/colamatch.git
cd colamatch
# git checkout develop
pip install -r requirements.txt
pip install -e .
```

## Usage
```python
import colamatch as clm
l_fixed = [p1,p2,...]  # landmarks in fixed images
l_moving = [p1,p2,...]  # landmarks in moving image
num_samples = 50000
sampler_fixed = clm.RandomSampler(len(l_fixed), 4, num_samples)
sampler_moving = clm.RandomSampler(len(l_moving), 4, num_samples)
matches = clm.match(l_fixed, l_moving, sampler_fixed, sampler_moving, radius=0.025, lamda=2, ransac=0.01)
```
