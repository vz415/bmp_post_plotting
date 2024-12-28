import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import Transform, constraints
from torch.distributions.transforms import SigmoidTransform, AffineTransform, ComposeTransform

# Define transformations
class InverseLogUniformBijector(Transform):
    domain = constraints.positive
    codomain = constraints.unit_interval

    def __init__(self, low, high):
        super().__init__()
        self.low = low
        self.high = high

    def __call__(self, y):
        log_y = torch.log(y)
        return (log_y - self.low) / (self.high - self.low)

    def _inverse(self, x):
        log_scaled = x * (self.high - self.low) + self.low
        return torch.exp(log_scaled)

    def log_abs_det_jacobian(self, y, x):
        return -torch.log(y * (self.high - self.low))

class LogUniformBijectorTorch(Transform):
    domain = constraints.unit_interval
    codomain = constraints.positive

    def __init__(self, low=1e-4, high=1e2):
        super().__init__()
        self.low = torch.log(torch.tensor(low).float())
        self.high = torch.log(torch.tensor(high).float())

    def __call__(self, x):
        log_scaled = x * (self.high - self.low) + self.low
        return torch.exp(log_scaled)

    def _inverse(self, y):
        log_y = torch.log(y)
        return (log_y - self.low) / (self.high - self.low)

    def log_abs_det_jacobian(self, x, y):
        return (self.high - self.low) + torch.log(x)

    @property
    def inv(self):
        return InverseLogUniformBijector(self.low, self.high)

def theta_transform_and_log_prob_adjustment_torch(shift, scale, low=1e-4, high=1e2):
    scalar_affine_transform = AffineTransform(loc=shift, scale=scale)
    expit_transform = SigmoidTransform()
    log_uniform_transform = LogUniformBijectorTorch(low=low, high=high)
    chained_transform = ComposeTransform([scalar_affine_transform, expit_transform, log_uniform_transform])
    return chained_transform

theta_z_score = theta_transform_and_log_prob_adjustment_torch(0, 1.81, low=1e-4, high=1e2)

# Load datasets (mocked here - replace with your file paths or data loaders)
@st.cache
def load_datasets():
    """
    Load posterior samples from .pth files into a dictionary format.

    Returns:
        dict: A dictionary mapping dataset names to their loaded data.
    """
    # Load data from .pth files
    post_lps_samples_all = torch.load('posterior_lps/post_lps_all_3_seeds.pth', map_location=torch.device('cpu'))
    post_lps_samples_0 = torch.load('posterior_lps/post_lps_0_3_seeds.pth', map_location=torch.device('cpu'))
    post_lps_samples_1 = torch.load('posterior_lps/post_lps_1_3_seeds.pth', map_location=torch.device('cpu'))
    post_lps_samples_2 = torch.load('posterior_lps/post_lps_2_3_seeds.pth', map_location=torch.device('cpu'))
    post_lps_samples_3 = torch.load('posterior_lps/post_lps_3_3_seeds.pth', map_location=torch.device('cpu'))

    # Organize datasets
    return {
        'All': post_lps_samples_all['seeds_data'][0],
        'NMuMG': post_lps_samples_0['seeds_data'][0],
        'BMPR2 KD (maybe ACVR1)': post_lps_samples_1['seeds_data'][0],
        'ACVR1 KD (maybe BMPR1A)': post_lps_samples_2['seeds_data'][0],
        'BMPR1A KD (maybe BMPR2)': post_lps_samples_3['seeds_data'][0],
    }


datasets = load_datasets()

# Z-score transformation
for key, dataset in datasets.items():
    raw_samples = torch.tensor(dataset['post_samples']).float()
    zscored = theta_z_score(raw_samples)
    dataset['zscored'] = zscored.numpy()

# Interactive Streamlit App
st.title("Interactive Plot for Posterior Samples")
st.sidebar.title("Controls")

# Sidebar options
current_dim = st.sidebar.slider("Select Dimension", 0, 59, 0)
selected_datasets = st.sidebar.multiselect(
    "Select Datasets", options=list(datasets.keys()), default=list(datasets.keys())
)

# Plot
fig, ax = plt.subplots(figsize=(7, 5))
bins = np.logspace(np.log10(1e-4), np.log10(1e2 + 1e-1), 50)

for dataset_name in selected_datasets:
    data = datasets[dataset_name]
    dim_samples = data['zscored'][:, current_dim]
    ax.hist(dim_samples, bins=bins, alpha=0.5, label=dataset_name, edgecolor='black', linewidth=0.5)

ax.set_xscale('log')
ax.set_xlabel(f"Dimension {current_dim}")
ax.set_ylabel("Counts")
ax.legend()

st.pyplot(fig)
