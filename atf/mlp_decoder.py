import torch
import torch.nn as nn

from atf.camo import WOODLAND_NUM_SUBCOLORS


class DualBranchDecoder(nn.Module):
    """MLP decoder with structure and category-aware color branches."""

    def __init__(self, input_dim=107, shared_hidden=256, branch_hidden=128,
                 num_categories=3, num_subcolors=WOODLAND_NUM_SUBCOLORS):
        super().__init__()
        self.num_categories = num_categories
        self.num_subcolors = num_subcolors

        self.shared = nn.Sequential(
            nn.Linear(input_dim, shared_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(shared_hidden, shared_hidden),
            nn.ReLU(inplace=True),
        )

        self.structure = nn.Sequential(
            nn.Linear(shared_hidden, branch_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(branch_hidden, 1),
            nn.Sigmoid(),
        )

        self.category = nn.Sequential(
            nn.Linear(shared_hidden, branch_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(branch_hidden, num_categories),
        )

        self.subcolor = nn.Sequential(
            nn.Linear(shared_hidden, branch_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(branch_hidden, branch_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(branch_hidden, num_categories * num_subcolors),
        )

        nn.init.constant_(self.structure[-2].bias, -2.0)

    def forward(self, feat):
        shared = self.shared(feat)
        alpha = self.structure(shared)
        category_logits = self.category(shared)
        subcolor_logits = self.subcolor(shared).view(
            -1, self.num_categories, self.num_subcolors)
        return alpha, category_logits, subcolor_logits
