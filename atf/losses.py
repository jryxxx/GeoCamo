"""ATF generalization-oriented regularization losses.

cross_vehicle_consistency_loss:  semantic alignment across vehicles
"""

import torch
import torch.nn.functional as F


def as_output_tuple(out):
    if torch.is_tensor(out):
        return (out,)
    return tuple(out)


def output_mse(out_a, out_b):
    out_a = as_output_tuple(out_a)
    out_b = as_output_tuple(out_b)
    return sum(F.mse_loss(a, b) for a, b in zip(out_a, out_b))


def cross_vehicle_consistency_loss(model, batch_geo_list):
    """Cross-vehicle geometric consistency.

    Points with similar geometric features across different vehicles should
    receive similar (alpha, c) outputs. This is the theoretical basis for
    ATF's zero-shot cross-vehicle transfer capability.

    Args:
        model: ATFModel
        batch_geo_list: list of (points_01, geo_features) tuples, one per vehicle

    Returns:
        scalar loss (0 for single vehicle)
    """
    n_vehicles = len(batch_geo_list)
    if n_vehicles < 2:
        return torch.tensor(0.0, device=batch_geo_list[0][0].device)

    total_loss = 0.0
    n_pairs = 0

    for i in range(n_vehicles):
        for j in range(i + 1, n_vehicles):
            pts_i, geo_i = batch_geo_list[i]
            pts_j, geo_j = batch_geo_list[j]

            geo_i_n = F.normalize(geo_i, dim=-1)
            geo_j_n = F.normalize(geo_j, dim=-1)

            dists = torch.cdist(geo_i_n, geo_j_n)
            nn_idx = dists.argmin(dim=-1)

            out_i = model(pts_i, geo_i)
            out_j = model(pts_j, geo_j)

            out_j = as_output_tuple(out_j)
            total_loss = total_loss + output_mse(
                out_i, tuple(component[nn_idx] for component in out_j))
            n_pairs += 1

    return total_loss / n_pairs
