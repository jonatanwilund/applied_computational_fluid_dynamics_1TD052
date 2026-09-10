import matplotlib.pyplot as plt
import numpy as np


def u0(
    x: np.ndarray, x0: np.ndarray | None = None, r0: float = 0.25
) -> np.ndarray:
    if x0 is None:
        x0 = np.array([0.3, 0])
    return 0.5 * (1 - np.tanh(((x[0] - x0[0]) ** 2 + (x[1] - x0[1]) ** 2) / r0**2 - 1))


def R(theta: float) -> np.ndarray:
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def u(x: np.ndarray, t: float) -> np.ndarray:
    # u(x, t) = u0(R(-2*pi*t) * x)
    R_inv = R(-2 * np.pi * t)
    x_orig = R_inv @ x
    return u0(x_orig)


def main():
    # Configure academic typography (Serif + LaTeX Computer Modern math)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Liberation Serif"],
            "mathtext.fontset": "cm",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    times = [0, 0.25, 0.5, 0.75]
    x_vals = np.linspace(-1, 1, 100)
    y_vals = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    mask = X**2 + Y**2 <= 1  # Unit disk mask
    U_masked = np.array([X[mask], Y[mask]])

    fig, axs = plt.subplots(
        2,
        2,
        subplot_kw={"projection": "3d"},
        figsize=(10, 9),
    )

    for ax, t in zip(axs.flat, times):
        Z_t = np.full_like(X, np.nan)
        # Evaluate exact solution at all grid points inside the disk
        Z_t[mask] = u(U_masked, t)

        surf = ax.plot_surface(
            X,
            Y,
            Z_t,
            cmap="viridis",
            vmin=0,
            vmax=1,
            rstride=1,
            cstride=1,
            linewidth=0,
            edgecolor="none",
            antialiased=False,
        )
        ax.set_title(f"$t = {t}$", fontsize=12)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_zlabel("$u$")
        ax.set_zlim(0, 1)
        ax.view_init(elev=35, azim=-60)

    # Shared colorbar across all 4 subplots
    fig.colorbar(surf, ax=axs, shrink=0.7, pad=0.08, label="$u(x, t)$")

    plt.savefig("project/part1/plots/u_over_time.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
