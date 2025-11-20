
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hex2color, LinearSegmentedColormap

BLUE = [hex2color(hex) for hex in ["#000000", "#CCCCCC", "#003A6D"]]
RED = [hex2color(hex) for hex in ["#000000", "#CCCCCC", "#9F1853"]]
blue = LinearSegmentedColormap.from_list('blue', BLUE, N=3)
red = LinearSegmentedColormap.from_list('red', RED, N=3)


def rgb_smooth_quantiles(array, tolerance=0.02, scaling=0.5, default=2000):
    """
    array: numpy array with dimensions [C, H, W]
    returns 0-1 scaled array
    """

    # Get scaling thresholds for smoothing the brightness
    limit_low, median, limit_high = np.quantile(array, q=[tolerance, 0.5, 1. - tolerance])
    limit_high = limit_high.clip(default)  # Scale only pixels above default value
    limit_low = limit_low.clip(0, 1000)  # Scale only pixels below 1000
    limit_low = np.where(median > default / 2, limit_low, 0)  # Make image only darker if it is not dark already

    # Smooth very dark and bright values using linear scaling
    array = np.where(array >= limit_low, array, limit_low + (array - limit_low) * scaling)
    array = np.where(array <= limit_high, array, limit_high + (array - limit_high) * scaling)

    # Update scaling params using a 10th of the tolerance for max value
    limit_low, limit_high = np.quantile(array, q=[tolerance/10, 1. - tolerance/10])
    limit_high = limit_high.clip(default, 20000)  # Scale only pixels above default value
    limit_low = limit_low.clip(0, 500)  # Scale only pixels below 500
    limit_low = np.where(median > default / 2, limit_low, 0)  # Make image only darker if it is not dark already

    # Scale data to 0-255
    array = (array - limit_low) / (limit_high - limit_low)

    return array


def s2_to_rgb(data, smooth_quantiles=False):
    if isinstance(data, torch.Tensor):
        # to numpy
        data = data.clone().cpu().numpy()
    if len(data.shape) == 4:
        # Remove batch dim
        return batch_s2_to_rgb(data, smooth_quantiles=smooth_quantiles)

    # Select
    if data.shape[0] > 13:
        # assuming channel last
        rgb = data[:, :, [3, 2, 1]]
    else:
        # assuming channel first
        rgb = data[[3, 2, 1]].transpose((1, 2, 0))

    if smooth_quantiles:
        rgb = rgb_smooth_quantiles(rgb)
    else:
        rgb = rgb / 2000

    # to uint8
    rgb = (rgb * 255).round().clip(0, 254).astype(np.uint8)

    return rgb


def batch_s2_to_rgb(data, smooth_quantiles=False):
    if isinstance(data, torch.Tensor):
        # to numpy
        data = data.clone().cpu().numpy()

    # assuming channel first
    rgb = data[:, [3, 2, 1]].transpose((0, 2, 3, 1))

    if smooth_quantiles:
        rgb = np.stack([rgb_smooth_quantiles(img) for img in rgb])
    else:
        rgb = rgb / 2000

    # to uint8
    rgb = (rgb * 255).round().clip(0, 255).astype(np.uint8)

    return rgb


def s1_to_rgb(data):
    if isinstance(data, torch.Tensor):
        # to numpy
        data = data.clone().cpu().numpy()
    if len(data.shape) == 4:
        # Remove batch dim
        return batch_s1_to_rgb(data)

    vv = data[0]
    vh = data[1]
    r = (vv + 30) / 40  # scale -30 to +10
    g = (vh + 40) / 40  # scale -40 to +0
    b = vv / np.nan_to_num(vh, nan=-40).clip(-40, -1) / 1.5  # VV / VH

    rgb = np.dstack([r, g, b])
    rgb = (rgb * 255).round().clip(0, 255).astype(np.uint8)
    return rgb


def batch_s1_to_rgb(data):
    if isinstance(data, torch.Tensor):
        # to numpy
        data = data.clone().cpu().numpy()

    vv = data[:, 0]
    vh = data[:, 1]
    r = (vv + 30) / 40  # scale -30 to +10
    g = (vh + 40) / 40  # scale -40 to +0
    b = vv / np.nan_to_num(vh, nan=-40).clip(-40, -1) / 1.5  # VV / VH

    rgb = np.stack([r, g, b], axis=-1)
    rgb = (rgb * 255).round().clip(0, 255).astype(np.uint8)
    return rgb


def s1_to_power(data):
    # Convert dB to power
    data = 10 ** (data / 10)
    return data * 10000


def s1_power_to_rgb(data):
    if isinstance(data, torch.Tensor):
        # to numpy
        data = data.clone().cpu().numpy()
    if len(data.shape) == 4:
        # Remove batch dim
        return batch_s1_power_to_rgb(data)

    data = np.nan_to_num(data, nan=-50)
    vv = data[0]
    vh = data[1]
    r = vv / 3000
    g = vh / 900
    b = vv / (vh + 1e-6) / 12

    rgb = np.dstack([r, g, b])
    rgb = (rgb * 255).round().clip(0, 255).astype(np.uint8)
    return rgb


def batch_s1_power_to_rgb(data):
    if isinstance(data, torch.Tensor):
        # to numpy
        data = data.clone().cpu().numpy()

    # data = np.nan_to_num(data, nan=-50)
    vv = data[:, 0]
    vh = data[:, 1]
    r = vv / 3000
    g = vh / 900
    b = vv / (np.nan_to_num(vh, nan=-40) + 1e-6) / 12

    rgb = np.dstack([r, g, b])
    rgb = (rgb * 255).round().clip(0, 255).astype(np.uint8)
    return rgb


def dem_to_rgb(data, cmap='BrBG_r', buffer=10):
    if isinstance(data, torch.Tensor):
        # to numpy
        data = data.clone().cpu().numpy()
    while len(data.shape) > 2:
        # Remove batch dim etc.
        return batch_dem_to_rgb(data, cmap=cmap, buffer=buffer)

    # Add 10m buffer to highlight flat areas
    data_min, data_max = np.nanmin(data), np.nanmin(data)
    data_min -= buffer
    data_max += buffer
    data = (data - data_min) / (data_max - data_min + 1e-6)

    rgb = plt.get_cmap(cmap)(data)[:, :, :3]
    rgb = (rgb * 255).round().clip(0, 255).astype(np.uint8)
    return rgb


def batch_dem_to_rgb(data, cmap='BrBG_r', buffer=10):
    if isinstance(data, torch.Tensor):
        # to numpy
        data = data.clone().cpu().numpy()

    # Add 10m buffer to highlight flat areas
    data_min, data_max = data.nanmin(axis=0), data.nanmax(axis=0)
    data_min -= buffer
    data_max += buffer
    data = (data - data_min) / (data_max - data_min + 1e-6)

    rgb = plt.get_cmap(cmap)(data)[:, :, :3]
    rgb = (rgb * 255).round().clip(0, 255).astype(np.uint8)
    return rgb


def mask_to_rgb(data, cmap=red, num_classes=2):
    while len(data.shape) > 2:
        if data.shape[0] == num_classes:
            data = data.argmax(axis=0)  # First dim are class logits
        else:
            # Remove batch dim
            data = data[0]

    rgb = cmap(data)[:, :, :3]
    rgb = (rgb * 255).round().clip(0, 255).astype(np.uint8)
    return rgb


def plot_s2(data, ax=None, smooth_quantiles=False, *args, **kwargs):
    rgb = s2_to_rgb(data, smooth_quantiles=smooth_quantiles)

    if ax is None:
        plt.imshow(rgb)
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(rgb)
        ax.axis('off')


def plot_s1(data, ax=None, power=False, *args, **kwargs):
    if power:
        data = s1_to_power(data)
        rgb = s1_power_to_rgb(data)
    else:
        rgb = s1_to_rgb(data)

    if ax is None:
        plt.imshow(rgb)
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(rgb)
        ax.axis('off')


def plot_dem(data, ax=None, *args, **kwargs):
    if isinstance(data, torch.Tensor):
        # to numpy
        data = data.clone().cpu().numpy()
    while len(data.shape) > 2:
        # Remove batch dim etc.
        data = data[0]

    # Add 10m buffer to highlight flat areas
    data_min, data_max = data.min(), data.max()
    data_min -= 5
    data_max += 5
    data = (data - data_min) / (data_max - data_min + 1e-6)

    data = (data * 255).round().clip(0, 255).astype(np.uint8)

    if ax is None:
        plt.imshow(data, vmin=0, vmax=255, cmap='BrBG_r')
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(data, vmin=0, vmax=255, cmap='BrBG_r')
        ax.axis('off')


def plot_mask(data, ax=None, num_classes=2, *args, **kwargs):
    if isinstance(data, torch.Tensor):
        # to numpy
        data = data.clone().cpu().numpy()
    while len(data.shape) > 2:
        if data.shape[0] == num_classes:
            data = data.argmax(axis=0)  # First dim are class logits
        else:
            # Remove batch dim
            data = data[0]

    if ax is None:
        plt.imshow(data, vmin=-1, vmax=num_classes-1, cmap=red, interpolation='nearest')
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(data, vmin=-1, vmax=num_classes-1, cmap=red, interpolation='nearest')
        ax.axis('off')


def plot_modality(modality, data, ax=None, **kwargs):
    if 's2' in modality.lower():
        plot_s2(data, ax=ax, **kwargs)
    elif 's1' in modality.lower():
        plot_s1(data, ax=ax, **kwargs)
    elif 'dem' in modality.lower():
        plot_dem(data, ax=ax, **kwargs)
    elif 'mask' in modality.lower():
        plot_mask(data, ax=ax, **kwargs)
