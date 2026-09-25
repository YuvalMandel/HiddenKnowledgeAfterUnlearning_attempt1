#!/usr/bin/env python3
"""Section 4.3 (Figures 4 and 9): K_int and K_ext across the unlearning checkpoints.

Prints, for every method, K_int and K_ext at each checkpoint (the values of
Figure 9; Figure 4 is the RMU row), then, at every checkpoint, their mean over
the eight methods, the mean gap and the range across methods: the numbers
quoted in Section 4.3. Checkpoint 0 is the shared base model.

Usage: python numbers/trajectories.py
"""
import numpy as np

from common import METHODS, N_CHECKPOINTS, label, load_k

XS = list(range(N_CHECKPOINTS + 1))


def mean_k(model_id: str) -> tuple[float, float]:
    d = load_k(model_id)
    return d.k_internal.mean(), d.k_external.mean()


def main():
    base = mean_k("base")
    ki, ke = [], []
    print(f"{'method':<10}{'':>6}" + "".join(f"{c:>7}" for c in XS))
    for m in METHODS:
        pairs = [base] + [mean_k(f"{m}_ck{c}") for c in XS[1:]]
        ki.append([p[0] for p in pairs])
        ke.append([p[1] for p in pairs])
        for name, row in (("K_int", ki[-1]), ("K_ext", ke[-1])):
            print(f"{label(m) if name == 'K_int' else '':<10}{name:>6}"
                  + "".join(f"{v:>7.3f}" for v in row))
    ki, ke = np.array(ki), np.array(ke)          # (method, checkpoint)

    print(f"\n{'checkpoint':<11}{'K_int':>7}{'K_ext':>7}{'gap pp':>8}"
          f"{'K_int range':>16}{'K_ext range':>15}")
    for c in XS:
        a, b = ki[:, c], ke[:, c]
        print(f"{'base' if c == 0 else c:<11}{a.mean():>7.3f}{b.mean():>7.3f}"
              f"{100 * (a.mean() - b.mean()):>8.1f}"
              f"{a.min():>10.3f}-{a.max():.3f}{b.min():>9.3f}-{b.max():.3f}")


if __name__ == "__main__":
    main()
