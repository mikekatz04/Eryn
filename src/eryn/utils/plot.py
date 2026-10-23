# *-- coding: utf-8 --*
"""Runtime diagnostic plots: :class:`PlotContainer` and the plotting functions behind it.

:class:`PlotContainer` reads a backend and writes PNG files under ``parent_folder``, one
sub-folder per plot group:

- ``base``: :func:`produce_base_plots` (per-branch cold-chain corner and trace plots,
  log-likelihood traces).
- ``tempering``: :func:`produce_tempering_plots` (ridge plots across temperatures, swap
  acceptance, ``<log L>`` against ``beta`` with a stepping-stone evidence, ``beta``
  evolution).
- ``advanced``: :func:`produce_advanced_plots` (acceptance-fraction history,
  autocorrelation-time evolution).
- ``rj``: :func:`produce_rj_plots` (histograms of the number of leaves).

:class:`~eryn.ensemble.EnsembleSampler` builds a default container (``base`` and ``rj``,
folder ``./runtime_plots``) when ``plot_iterations > 0`` and no ``plot_generator`` is
given, and calls :meth:`PlotContainer.produce_plots` every ``plot_iterations`` steps.

Importing this module (which :mod:`eryn.utils` and :mod:`eryn.ensemble` do) imports
corner, matplotlib, pandas and seaborn, applies the ``scienceplots`` ``"science"`` style
when that package is installed, and sets the global matplotlib font size to 16.
"""
import os
from shutil import which
import typing

import corner
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import to_rgba
from matplotlib.patches import Ellipse, Rectangle

from eryn.utils.updates import UpdateStep
from eryn.utils.utility import get_integrated_act, stepping_stone_log_evidence

DEFAULT_PALETTE = "icefire"

try:
    import scienceplots
    plt.style.use(['science'])
except (ImportError, ModuleNotFoundError, AttributeError):
    pass

# increase default font size
mpl.rcParams.update({'font.size': 16})

class Backend:
    """A placeholder Backend class for type hinting."""
    pass

def setup_plotting():
    """Set the global matplotlib style used by the diagnostic plots.

    Applies the ``scienceplots`` ``"science"`` style when available, sets the font size to
    16, and enables LaTeX text with a Palatino serif font when a ``latex`` executable is on
    ``PATH``; otherwise prints a notice and disables ``text.usetex``. Called by
    :class:`PlotContainer` on construction. Changes ``matplotlib.rcParams`` globally.
    """
    try:
        import scienceplots
        plt.style.use(['science'])
    except:
        pass

    # increase default font size
    mpl.rcParams.update({'font.size': 16})

    if which("latex"):
        mpl.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Palatino"],
        })
    else:
        print("LaTeX not found. Using default matplotlib fonts.")
        mpl.rcParams.update({
            "text.usetex": False,
        })


def save_or_show(fig, filename=None):
    """Save ``fig`` to ``filename`` (200 dpi, tight bounding box) and close it, or show it.

    Args:
        fig (matplotlib.figure.Figure): Figure to save or show.
        filename (str, optional): Output path. Default ``None``: call ``plt.show()``
            instead (the figure is not closed).
    """
    if filename:
        fig.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

def cov_ellipse(mean, cov, ax, n_std=1.0, **kwargs):
    """Draw the ``n_std`` covariance ellipse of a 2-D Gaussian on ``ax``.

    The ellipse axes follow the eigenvectors of ``cov``, with semi-axes
    ``n_std * sqrt(eigenvalue)``.

    Args:
        mean (array_like): Centre ``(x, y)``.
        cov (np.ndarray): 2x2 covariance matrix.
        ax (matplotlib.axes.Axes): Axes to draw on.
        n_std (float, optional): Number of standard deviations. Default ``1.0``.
        **kwargs: Forwarded to :class:`matplotlib.patches.Ellipse`.

    Returns:
        matplotlib.patches.Ellipse: The patch added to ``ax``.
    """
    # Eigendecomposition: eigenvalues are variances along principal axes
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Sort by eigenvalue (largest first) for consistent orientation
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    # Ellipse dimensions: 2 * n_std * sqrt(eigenvalue) for width/height
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    
    # Rotation angle from the first eigenvector (major axis direction)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    return ax.add_patch(ellipse)

def overlay_fim_covariance(
    fig,
    covariance,
    means=None,
    nsigmas=[1, 2, 3],
    plot_1d=False,
    colors=None,
    linestyles=None,
    linewidths=None,
    alpha=0.7,
    labels=None,
):
    """Overlay Gaussian (e.g. Fisher-matrix) confidence contours on a corner plot.

    The figure's axes are taken as an ``n_params x n_params`` corner grid. Off-diagonal
    panels get one :func:`cov_ellipse` per sigma level; with ``plot_1d`` the diagonal panels
    get vertical lines at ``mean +- n * sigma``.

    Args:
        fig (matplotlib.figure.Figure): Figure from ``corner.corner``.
        covariance (np.ndarray): Covariance matrix, shape ``(n_params, n_params)``.
        means (np.ndarray, optional): Centre of the contours, length ``n_params``. Default
            ``None`` (zeros).
        nsigmas (list, optional): Sigma levels. Default ``[1, 2, 3]``.
        plot_1d (bool, optional): Also draw the 1-D lines on the diagonal. Default
            ``False``.
        colors (list or str, optional): One colour per level, or one colour for all.
            Default ``None`` (``"C0"``, ``"C1"``, ...).
        linestyles (list, optional): One line style per level. Default ``None`` (solid).
        linewidths (list, optional): One line width per level. Default ``None`` (1.5).
        alpha (float, optional): Contour transparency. Default ``0.7``.
        labels (list, optional): One label per level. Default ``None`` (an "n sigma FIM"
            LaTeX label per level). Only validated: no legend is drawn here.

    Returns:
        matplotlib.figure.Figure: ``fig``, with the contours added.

    Raises:
        ValueError: If ``covariance`` is not square or a per-level list (or ``means``) has
            the wrong length.
    """
    
    # Convert axs to numpy array if it's a list
    axs = np.array(fig.get_axes())
    
    # Infer number of parameters from covariance matrix
    n_params = covariance.shape[0]
    
    if covariance.shape != (n_params, n_params):
        raise ValueError(f"Covariance matrix must be square, got shape {covariance.shape}")
    
    # Set default means to origin
    if means is None:
        means = np.zeros(n_params)
    elif len(means) != n_params:
        raise ValueError(f"means must have length {n_params}, got {len(means)}")
    
    # Set default colors
    if colors is None:
        colors = [f'C{i}' for i in range(len(nsigmas))]
    elif isinstance(colors, str):
        colors = [colors] * len(nsigmas)
    elif len(colors) != len(nsigmas):
        raise ValueError(f"colors must have length {len(nsigmas)}, got {len(colors)}")
    
    # Set default linestyles
    if linestyles is None:
        linestyles = ['-'] * len(nsigmas)
    elif len(linestyles) != len(nsigmas):
        raise ValueError(f"linestyles must have length {len(nsigmas)}, got {len(linestyles)}")
    
    # Set default linewidths
    if linewidths is None:
        linewidths = [1.5] * len(nsigmas)
    elif len(linewidths) != len(nsigmas):
        raise ValueError(f"linewidths must have length {len(nsigmas)}, got {len(linewidths)}")
    
    # Set default labels
    if labels is None:
        labels = [f'{n}$\\sigma$ FIM' for n in nsigmas]
    elif len(labels) != len(nsigmas):
        raise ValueError(f"labels must have length {len(nsigmas)}, got {len(labels)}")
    
    # Extract standard deviations for 1D plots
    sigmas = np.sqrt(np.diag(covariance))
    
    # Reshape axes into 2D grid if needed
    if axs.ndim == 1:
        # Corner plot axes are typically returned as 1D array
        # Reshape to (n_params, n_params) grid
        n_axs = int(np.sqrt(len(axs)))
        axs_grid = np.empty((n_axs, n_axs), dtype=object)
        idx = 0
        for i in range(n_axs):
            for j in range(n_axs):
                axs_grid[j, i] = axs[idx]
                idx += 1
                # if j <= i:
                #     axs_grid[j, i] = axs[idx]
                #     idx += 1
                # else:
                #     axs_grid[j, i] = None
    else:
        axs_grid = axs
    
    # Loop over each sigma level
    for sigma_idx, (n_sigma, color, ls, lw, label) in enumerate(
        zip(nsigmas, colors, linestyles, linewidths, labels)
    ):        
        # Loop over all subplots
        for i in range(n_params):
            for j in range(i, n_params):
                ax = axs_grid[i, j]
                
                if ax is None:
                    continue
                
                if i == j:
                    if plot_1d:
                        # 1D diagonal plot - draw vertical lines at mean ± n*sigma
                        mean_val = means[i]
                        sigma_val = sigmas[i]
                        
                        # Get y-limits for vertical lines
                        ylim = ax.get_ylim()
                        
                        # Draw vertical lines
                        for sign in [-1, 1]:
                            line = ax.axvline(
                                mean_val + sign * n_sigma * sigma_val,
                                color=color,
                                linestyle=ls,
                                linewidth=lw,
                                alpha=alpha,
                                zorder=10,
                            )
                        else:
                            continue
                
                else:
                    # 2D off-diagonal plot - draw ellipse
                    # Extract 2x2 subcovariance for parameters j and i
                    cov = np.array(
                    (
                        (covariance[i][i], covariance[i][j]),
                        (covariance[j][i], covariance[j][j]),
                    )
                )
                # print(cov)

                    mean = np.array((means[i], means[j]))
                    cov_ellipse(mean, cov, ax, n_std=n_sigma, 
                                edgecolor=color, facecolor='none', linestyle=ls, linewidth=lw, 
                                zorder=10, alpha=alpha
                                )
    
    return fig

def cornerplot(data, *args, means=None, overlay_covariance=None, legend_label='Samples', overlay_label='Information Matrix Covariance', filename=None, **kwargs):
    """Corner plot of samples, with optional truths and covariance-contour overlay.

    Wraps ``corner.corner`` with default settings (1, 2 and 3 sigma 2-D levels, filled
    contours, titles, density histograms, ``color="steelblue"``, ``truth_color="red"``),
    which ``**kwargs`` override. A 2-D ``truths`` array (several truth points, e.g. one per
    leaf of a reversible-jump branch) is drawn as lines on every panel instead of being
    passed to corner. A legend (samples, truths, overlay) goes on the second axis, and the
    figure is saved or shown with :func:`save_or_show`.

    Args:
        data (np.ndarray): Samples, shape ``(nsamples, ndim)`` with ``ndim >= 2``.
        *args: Forwarded to ``corner.corner``.
        means (array_like, optional): Centre of the overlay. Default ``None``: the
            ``truths`` keyword, else the sample mean.
        overlay_covariance (np.ndarray, optional): Covariance to overlay with
            :func:`overlay_fim_covariance` (1, 2, 3 sigma, black). Default ``None`` (none).
        legend_label (str, optional): Legend label of the samples. Default ``"Samples"``.
        overlay_label (str, optional): Legend label of the overlay. Default
            ``"Information Matrix Covariance"``.
        filename (str, optional): Output path. Default ``None`` (show).
        **kwargs: Forwarded to ``corner.corner`` (e.g. ``labels``, ``truths``,
            ``color``); ``truths`` must be an ``np.ndarray`` if given.

    Returns:
        None: The figure is saved (and closed) or shown.
    """

    corner_kwargs = {
    #'quantiles': [0.16, 0.5, 0.84],
    'levels': (1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-4.5)),
    'show_titles': True,
    'title_fmt': '.1e',
    'title_kwargs': {'fontsize': 12},
    'label_kwargs': {'fontsize': 14},
    'hist_kwargs': {'density': True, 'linewidth': 2},
    'plot_datapoints': False,
    'fill_contours': True,
    'color': 'steelblue',
    'truth_color': 'red',
    }

    # Update base kwargs with any user-provided kwargs
    corner_kwargs.update(kwargs)

    # how do we deal with 2d truths for reversible jump?
    truths = corner_kwargs.get('truths', None)
    if truths is not None and len(truths.shape) > 1:  
        corner_kwargs['truths'] = None # add truths later

    fig = corner.corner(data=data, *args, **corner_kwargs)

    # add the other truths as vertical/horizontal lines
    if truths is not None and len(truths.shape) > 1:  
        corner_kwargs['truths'] = truths # add back in for legend
        n_params = data.shape[1]
        axs = np.array(fig.get_axes()).reshape((n_params, n_params))
        for truth in truths:
            for i in range(n_params):
                ax = axs[i, i]
                ax.axvline(truth[i], color=corner_kwargs['truth_color'], linestyle='-', linewidth=1)

            for i in range(n_params):
                for j in range(i):
                    ax = axs[i, j]
                    ax.axhline(truth[i], color=corner_kwargs['truth_color'], linestyle='-', linewidth=1)
                    ax.axvline(truth[j], color=corner_kwargs['truth_color'], linestyle='-', linewidth=1)

    # prepare handles for legend
    handles = []
    handles_labels = []

    handles.append(mpl.lines.Line2D([], [], color=corner_kwargs['color'],
                                    linestyle='-', linewidth=2))
    handles_labels.append(legend_label)
    if 'truths' in corner_kwargs and corner_kwargs['truths'] is not None:
        handles.append(mpl.lines.Line2D([], [], color=corner_kwargs['truth_color'],
                                        linestyle='-', linewidth=1))
        handles_labels.append('Truths')


    # Overlay covariance contours if provided
    if overlay_covariance is not None:
        if means is None:
            means = kwargs.get('truths', None)
        
        if means is None:
            # If no means or truths provided, default to the mean of the samples
            means = np.mean(data, axis=0)

        overlay_fim_covariance(
            fig,
            overlay_covariance,
            means=means,
            plot_1d=False,
            alpha=0.8,
            nsigmas=[1, 2 ,3],   
            colors='k'
        )

        handles.append(mpl.lines.Line2D([], [], color='k', linestyle='-', linewidth=1.5))
        handles_labels.append(overlay_label)
    
    # Add legend to the second leftmost top subplot
    ax_legend = fig.get_axes()[1]
    ax_legend.legend(handles, handles_labels, loc='upper left', fontsize=18)
    
    save_or_show(fig, filename)


def traceplot(chain, labels=None, truths=None, filename=None):
    """Trace plots: one panel per parameter, every walker and leaf against step.

    Args:
        chain (np.ndarray): Chain of shape ``(nsteps, nwalkers, nleaves, ndim)`` with
            ``ndim >= 2`` (a single panel is not indexable).
        labels (list, optional): Per-parameter y-axis labels. Default ``None``.
        truths (array_like, optional): Truth values, shape ``(ndim,)`` or
            ``(ntruths, ndim)``, drawn as dashed horizontal lines. Default ``None``.
        filename (str, optional): Output path. Default ``None`` (show).

    Returns:
        None: The figure is saved (and closed) or shown.
    """

    nsteps, nwalkers, nleaves, ndim = chain.shape
    fig, axs = plt.subplots(ndim, 1, figsize=(10, 2.5 * ndim), sharex=True)
    
    for i in range(ndim):
        for w in range(nwalkers):
            axs[i].plot(chain[:, w, :, i], alpha=0.5, rasterized=True)
            if truths is not None:
                truths = np.atleast_2d(truths)
                for j in range(truths.shape[0]):
                    axs[i].axhline(truths[j, i], color='k', linestyle='--')
            if labels is not None:
                axs[i].set_ylabel(labels[i])

    axs[-1].set_xlabel('Step')
    plt.tight_layout()

    save_or_show(fig, filename)


def plot_loglikelihood(logl, filename=None):
    """Plot log-likelihood traces per walker, plus a per-walker facet grid.

    The first figure has one line per walker. The second (a seaborn ``FacetGrid``, one
    panel per walker) shows ``logl - max(logl)`` and is saved to ``filename`` with
    ``.png`` replaced by ``_facet.png``; any failure while building it is logged as a
    warning and skipped, so it never aborts a run.

    Args:
        logl (np.ndarray): Log-likelihood, shape ``(nsteps, nwalkers)`` (e.g. the cold
            chain).
        filename (str, optional): Output path of the first figure. Default ``None``
            (show both).

    Returns:
        None: The figures are saved (and closed) or shown.
    """
    nsteps, nwalkers = logl.shape
    fig = plt.figure(figsize=(10, 6))
    for j in range(nwalkers):
        plt.plot(logl[:, j], color=f"C{j % 10}", alpha=0.8, rasterized=True)
    plt.xlabel("Sampler Iteration")
    plt.ylabel("Log-Likelihood")
    
    save_or_show(fig, filename)

    # plot a facet grid of loglikelihood evolution for each walker.
    # This is a *diagnostic* and must never abort a (multi-hour) run: with few
    # walkers/steps and large per-walker logL offsets the small facets can make
    # matplotlib's tight_layout produce a NaN axis length ("cannot convert float
    # NaN to integer"). Guard the whole block so a layout failure is skipped, not
    # fatal; the primary log-likelihood plot above is already saved.
    grid = None
    try:
        # Reshape: logl is (nsteps, nwalkers), need to flatten properly
        max_logl = np.max(logl, axis=(0,1))
        facet_logl = logl - max_logl

        step = np.tile(range(nsteps), nwalkers)
        walker = np.repeat(np.arange(nwalkers, dtype=int), nsteps)

        df = pd.DataFrame(np.c_[facet_logl.flat, step, walker],
                          columns=[r"$\Delta \log\mathcal{L}$", "step", "walker"])

        # Initialize a grid of plots with an Axes for each walker
        grid = sns.FacetGrid(df, col="walker", hue="walker", #palette="tab20c",
                         col_wrap=int(np.floor(np.sqrt(nwalkers))), height=1.5)

        # Draw a line plot to show the trajectory of each random walk
        grid.map(plt.plot, "step", r"$\Delta \log\mathcal{L}$", marker=".", rasterized=True)

        grid.refline(y=0, linestyle=":") # Add a horizontal reference line at y=0 ~ average loglikelihood at each step

        # Adjust the arrangement of the plots
        grid.set_titles(col_template="Walker {col_name:.0f}")
        grid.set_axis_labels("Step", r"$\Delta \log\mathcal{L}$")
        # Disable tight_layout to avoid warning
        grid.tight_layout = lambda *args, **kwargs: None
        grid.tight_layout()

        # add overall title
        plt.subplots_adjust(top=0.9)
        grid.figure.suptitle(r"$\Delta \log\mathcal{L}_w = \log\mathcal{L}_w - \max(\log\mathcal{L})$", fontsize=16)

        save_or_show(grid.figure, filename.replace('.png', '_facet.png') if filename else None)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            "Skipping per-walker log-likelihood facet plot: %s: %s", type(e).__name__, e
        )
        if grid is not None:
            plt.close(grid.figure)

def tempering_ridgeplot(chain, labels=None, palette=None, 
                        bw_adjust=0.5, aspect=5, height=0.5, hspace=-0.25,
                        max_samples=10000, filename=None):
    """Ridge (joy) plot of each parameter's marginal at every temperature.

    One seaborn ``FacetGrid`` with a row per temperature (cold at the top, rows labelled
    by index ``beta_t``) and a column per parameter; each panel is a filled KDE of all
    steps, walkers and leaves at that temperature (NaNs dropped). Each column's x-range is
    the cold chain's 0.1-99.9 % quantile range plus a 30 % margin.

    Args:
        chain (np.ndarray): Chain of shape ``(nsteps, ntemps, nwalkers, nleaves, ndim)``.
        labels (list, optional): Per-parameter labels for the x axes. Default ``None``
            (``$x_i$``).
        palette (str or list, optional): Seaborn palette name or list of colours, one per
            temperature. Default ``None`` (``DEFAULT_PALETTE = "icefire"``).
        bw_adjust (float, optional): KDE bandwidth factor. Default ``0.5``.
        aspect (float, optional): Facet aspect ratio. Default ``5``.
        height (float, optional): Height of each row in inches. Default ``0.5``.
        hspace (float, optional): Vertical spacing between rows (negative overlaps them).
            Default ``-0.25``.
        max_samples (int, optional): Per temperature and parameter, subsample to at most
            this many points (fixed seed 42) before the KDE. Default ``10000``.
        filename (str, optional): Output path. Default ``None`` (show).

    Returns:
        None: The figure is saved (and closed) or shown.
    """
    # Use seaborn context manager to temporarily set theme
    with sns.axes_style("white", {"axes.facecolor": (0, 0, 0, 0)}):

        nsteps, ntemps, nwalkers, nleaves, ndim = chain.shape
        
        # Create color palette (blue=cold/β=1 at top, red=hot/β→0 at bottom)
        if palette is None:
            # Use coolwarm reversed: blue for cold (β=1), red for hot (β→0)
            pal = sns.color_palette(DEFAULT_PALETTE, ntemps)
        elif isinstance(palette, str):
            pal = sns.color_palette(palette, ntemps)
        else:
            pal = palette
        
        # Subsampling RNG
        rng = np.random.default_rng(42)
        
        # Build dataframe with samples from all temperatures and parameters
        data_list = []
        for param_idx in range(ndim):
            param_label = labels[param_idx] if labels is not None else fr'$x_{param_idx}$'
            for t in range(ntemps):
                # Flatten samples across steps, walkers, leaves for the selected parameter
                samples = chain[:, t, :, :, param_idx].reshape(-1)
                
                # Remove NaNs
                samples = samples[~np.isnan(samples)]
                
                # Subsample if needed for faster KDE
                if len(samples) > max_samples:
                    samples = rng.choice(samples, size=max_samples, replace=False)
                
                temp_label = rf"$\beta_{{{t}}}$" #fr"$\beta$={betas[t]:.1e}"
                data_list.append(pd.DataFrame({
                    'x': samples,
                    'temp': temp_label,
                    'temp_idx': t,
                    'param': param_label,
                    'param_idx': param_idx
                }))
        
        df = pd.concat(data_list, ignore_index=True)
        
        # Get unique temps in order (beta=1 first at top)
        temp_order = df.drop_duplicates('temp_idx').sort_values('temp_idx', ascending=True)['temp'].tolist()
        
        # Get unique params in order
        param_order = df.drop_duplicates('param_idx').sort_values('param_idx', ascending=True)['param'].tolist()
        
        # Compute x-axis limits from the cold posterior (β=1, temp_idx=0) for each parameter
        # This ensures the cold posterior is always visible even when hot distributions are much wider
        xlims = {}
        cold_df = df[df['temp_idx'] == 0]
        for param in param_order:
            param_data = cold_df[cold_df['param'] == param]['x']
            q_low, q_high = param_data.quantile([0.001, 0.999])
            margin = (q_high - q_low) * 0.3  # Add 30% margin to show some broadening
            xlims[param] = (q_low - margin, q_high + margin)
        
        # Initialize the FacetGrid with row=temp, col=param
        # Suppress tight_layout warnings during initialization
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Tight layout.*")
            g = sns.FacetGrid(df, row="temp", col="param", hue="temp", 
                            aspect=aspect, height=height, 
                            palette=pal, row_order=temp_order, col_order=param_order,
                            sharex=False, sharey=False)
        
        # Disable tight_layout to avoid warning with negative hspace
        g.tight_layout = lambda *args, **kwargs: None
        
        # Custom plotting function that clips KDE to the parameter's xlim
        def plot_kde_clipped(x, color, label, **kwargs):
            ax = plt.gca()
            # Get the parameter for this column
            col_idx = ax.get_subplotspec().colspan.start
            param = param_order[col_idx]
            clip = xlims[param]
            
            sns.kdeplot(x, ax=ax, bw_adjust=bw_adjust, clip_on=False,
                        fill=True, alpha=1, linewidth=1.5, color=color, clip=clip)
            sns.kdeplot(x, ax=ax, clip_on=False, color="w", lw=2, bw_adjust=bw_adjust, clip=clip)
            ax.set_xlim(clip)
        
        g.map(plot_kde_clipped, "x")
        
        # Add horizontal reference line at y=0
        g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
        
        # Label function for temperature labels (only on first column)
        def label_temp(x, color, label):
            ax = plt.gca()
            # Only add label if this is the first column
            if ax.get_subplotspec().colspan.start == 0:
                ax.text(-0.2, 0.2, label, fontweight="bold", color=color,
                        ha="left", va="center", transform=ax.transAxes, fontsize=14)
        
        g.map(label_temp, "x")
        
        # Set subplots to overlap vertically
        g.figure.subplots_adjust(hspace=hspace, wspace=0.1)

        # use parameter labels in the x axes (on the last row)
        for ax, param in zip(g.axes[-1], param_order):
            ax.set_xlabel(param, fontsize=14)

        # Remove xticks from all rows except the bottom
        for row_idx in range(len(temp_order) - 1):
            for col_idx in range(len(param_order)):
                g.axes[row_idx, col_idx].set_xticks([])

                # set xlims again to ensure consistency
                param = param_order[col_idx]
                g.axes[row_idx, col_idx].set_xlim(xlims[param])
        
        # Remove axes details that don't work well with overlap
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)
        
        # Add overall title
        g.figure.suptitle('Tempered Distributions', y=1.02, fontsize=14)
        
        save_or_show(g.figure, filename)

    # # Properly restore the original seaborn style
    # sns.set_theme(style=original_style['style'], rc=original_style)
    
    # return g.figure

def plot_swap_acceptance(swap_acceptance_fraction, palette=None, filename=None):
    """Bar chart of the swap acceptance fraction of each adjacent temperature pair.

    Bars are annotated with their values; a dashed reference line marks 0.25.

    Args:
        swap_acceptance_fraction (np.ndarray): Shape ``(ntemps - 1,)``; element ``i`` is the
            acceptance of swaps between temperatures ``i`` and ``i + 1``.
        palette (str or list, optional): Seaborn palette name or list of colours. Default
            ``None`` (``"icefire"``).
        filename (str, optional): Output path. Default ``None`` (show).

    Returns:
        matplotlib.figure.Figure: The figure (already closed when ``filename`` is given).
    """
    ntemps = swap_acceptance_fraction.shape[0] + 1
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # X-axis: temperature pair indices (swap between temp i and i+1)
    x = np.arange(ntemps - 1)
    
    # Create labels for each swap pair
    labels = [fr'{i}$\leftrightarrow${i+1}' for i in range(ntemps - 1)]
    
    # Color by temperature (use coolwarm to match ridgeplot)
    palette = palette if palette is not None else DEFAULT_PALETTE
    colors = sns.color_palette(palette, ntemps - 1)
    
    # Bar plot
    bars = ax.bar(x, swap_acceptance_fraction, color=colors, edgecolor='white', linewidth=0.5)
    
    # Add horizontal reference line at common target acceptance (0.2-0.4 is often good)
    ax.axhline(y=0.25, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='0.25')
    
    # Add beta values as secondary labels
    # ax2 = ax.twiny()
    # ax2.set_xlim(ax.get_xlim())
    # ax2.set_xticks(x)
    # beta_labels = [fr'$\beta$={betas[i]:.1e}$\leftrightarrow${betas[i+1]:.1e}' for i in range(ntemps - 1)]
    # ax2.set_xticklabels(beta_labels, fontsize=8, rotation=45, ha='left')
    
    # Labels and formatting
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=45, ha='right')
    ax.set_xlabel('Temperature Pair (index)', fontsize=12)
    ax.set_ylabel('Swap Acceptance Fraction', fontsize=12)
    ax.set_title('Temperature Swap Acceptance', fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper left', fontsize=10)
    
    # Add value annotations on bars
    for bar, val in zip(bars, swap_acceptance_fraction):
        height = bar.get_height()
        ax.annotate(f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    save_or_show(fig, filename)
    
    return fig
    
def plot_logl_betas(betas: np.ndarray, 
                    logl: np.ndarray, 
                    palette: str = None,
                    filename: str = None
                    ):
        """Plot the mean log-likelihood against ``beta``, titled with the evidence.

        One point per temperature at ``(betas[-1, t], mean(logl[:, t]))`` (x axis log
        scale); the title shows ``log Z +- dlog Z`` from
        :func:`~eryn.utils.utility.stepping_stone_log_evidence` with the last step's
        ``betas``.

        Args:
            betas: Inverse temperatures, shape ``(nsteps, ntemps)``.
            logl: Log-likelihood, shape ``(nsteps, ntemps, nwalkers)``.
            palette: Seaborn palette name or list of colours. Default ``None``
                (``"icefire"``).
            filename: Output path. Default ``None`` (show).

        Returns:
            None: The figure is saved (and closed) or shown.
        """
        fig = plt.figure(figsize=(10, 6))
        ntemp = betas.shape[1]
        tempcolors = sns.color_palette(palette if palette is not None else DEFAULT_PALETTE, ntemp)
        for temp in range(ntemp):
            plt.semilogx(betas[-1, temp], np.mean(logl[:, temp]), '.', c=tempcolors[temp], label=f'$T_{temp}$')

        logZ, dlogZ = stepping_stone_log_evidence(betas[-1], logl)
        
        plt.ylabel(r'$<\log{\mathcal{L}}>_{\beta}$')
        plt.xlabel(r'$\beta$')
        plt.title(r'$\log{\mathcal{Z}} = %.2f \pm %.2f$' % (logZ, dlogZ))
        
        save_or_show(fig, filename)

def plot_betas_evolution(betas: np.ndarray, palette: str = None, filename: str = None):
    """Plot every inverse temperature against sampler step.

    Args:
        betas: Inverse temperatures, shape ``(nsteps, ntemps)``.
        palette: Seaborn palette name or list of colours. Default ``None`` (``"icefire"``).
        filename: Output path. Default ``None`` (show).

    Returns:
        None: The figure is saved (and closed) or shown.
    """
    nsteps, ntemps = betas.shape
    tempcolors = sns.color_palette(palette if palette is not None else DEFAULT_PALETTE, ntemps)
    fig = plt.figure(figsize=(10, 6))
    for temp in range(ntemps):
        plt.plot(range(nsteps), betas[:, temp], color=tempcolors[temp], linewidth=1.5, alpha=0.8, rasterized=True)
    plt.xlabel('Sampler Iteration')
    plt.ylabel(r'Inverse Temperature ($\beta$)')
    plt.title('Evolution of Inverse Temperatures')
    
    # Create temperature color gradient legend
    
    ax = plt.gca()
    legend_width = 0.15
    legend_height = 0.03
    legend_x = 0.75
    legend_y = 0.9
    
    # Create gradient patches
    for i, color in enumerate(tempcolors[::-1]):  # Reverse to show cold->hot left to right
        rect = Rectangle(
            (legend_x + i * legend_width / ntemps, legend_y),
            legend_width / ntemps, legend_height,
            transform=ax.transAxes,
            facecolor=to_rgba(color, 0.7),
            edgecolor='none'
        )
        ax.add_patch(rect)
    
    # Add border and labels
    border = Rectangle(
        (legend_x, legend_y), legend_width, legend_height,
        transform=ax.transAxes,
        facecolor='none',
        edgecolor='black',
        linewidth=0.5
    )
    ax.add_patch(border)
    
    # Add text labels
    ax.text(legend_x - 0.01, legend_y + legend_height / 2, r'$T_{\rm max}$',
            transform=ax.transAxes, ha='right', va='center', fontsize=11, fontweight='normal', antialiased=True)
    ax.text(legend_x + legend_width + 0.01, legend_y + legend_height / 2, r'$T_0$',
            transform=ax.transAxes, ha='left', va='center', fontsize=11, fontweight='normal', antialiased=True)
    
    save_or_show(fig, filename)


# RJ plots
def plot_leaves(nleaves: np.ndarray,
                nleaves_min: int,
                nleaves_max: int,
                palette: str = None,
                iteration: int = 0,
                filename: str = None):
    """Histogram of the number of leaves at each temperature, for one branch.

    One step-filled, density-normalised histogram per temperature with integer-centred bins
    from ``nleaves_min`` to ``nleaves_max``, a colour-gradient legend from ``T_0`` to
    ``T_max``, and the step number printed in the corner.

    Args:
        nleaves: Leaf counts, shape ``(nsteps, ntemps, nwalkers)``.
        nleaves_min: Smallest possible number of leaves.
        nleaves_max: Largest possible number of leaves.
        palette: Seaborn palette name or list of colours. Default ``None`` (``"icefire"``).
        iteration: Step number shown on the figure. Default ``0``.
        filename: Output path. Default ``None`` (show).

    Returns:
        None: The figure is saved (and closed) or shown.
    """
    bns = (np.arange(nleaves_min, nleaves_max + 2) - 0.5)
    ntemps = nleaves.shape[1]
    tempcolors = sns.color_palette(palette if palette is not None else DEFAULT_PALETTE, ntemps)

    fig = plt.figure(figsize=(8, 5))

    for temp, tempcolor in enumerate(tempcolors):
        plt.hist(nleaves[:, temp].flatten(), bins=bns, histtype="stepfilled", edgecolor=tempcolor,
                    facecolor=to_rgba(tempcolor, 0.2), density=True, ls='-', zorder=100 - temp, rasterized=True)

    plt.xlabel('Number of leaves')
    plt.ylabel('Density')
    
    # Create temperature color gradient legend
    # Use a horizontal gradient showing cold (blue) to hot (red)
    
    # Add a color bar as legend showing temperature progression
    ax = plt.gca()
    legend_width = 0.15
    legend_height = 0.03
    legend_x = 0.75
    legend_y = 0.9
    
    # Create gradient patches
    n_gradient = len(tempcolors)
    for i, color in enumerate(tempcolors[::-1]):  # Reverse to show cold->hot left to right
        rect = Rectangle(
            (legend_x + i * legend_width / n_gradient, legend_y),
            legend_width / n_gradient, legend_height,
            transform=ax.transAxes,
            facecolor=to_rgba(color, 0.7),
            edgecolor='none'
        )
        ax.add_patch(rect)
    
    # Add border and labels
    border = Rectangle(
        (legend_x, legend_y), legend_width, legend_height,
        transform=ax.transAxes,
        facecolor='none',
        edgecolor='black',
        linewidth=0.5
    )
    ax.add_patch(border)
    
    # Add text labels
    ax.text(legend_x - 0.01, legend_y + legend_height / 2, r'$T_{\rm max}$',
            transform=ax.transAxes, ha='right', va='center', fontsize=11, fontweight='normal', antialiased=True)
    ax.text(legend_x + legend_width + 0.01, legend_y + legend_height / 2, r'$T_0$',
            transform=ax.transAxes, ha='left', va='center', fontsize=11, fontweight='normal', antialiased=True)

    fig.text(0.07, 0.08, f"Step: {iteration}", ha='left', va='top', fontfamily='serif', c='k')
    #plt.title(key)
    save_or_show(fig, filename)

def plot_leaves_evolution(nleaves: np.ndarray,
                          filename: str = None):
    """Plot the number of leaves of each cold-chain walker against sampler step.

    Not called by :func:`produce_rj_plots` (its call there is commented out).

    Args:
        nleaves: Cold-chain leaf counts, shape ``(nsteps, nwalkers)``.
        filename: Output path. Default ``None`` (show).

    Returns:
        None: The figure is saved (and closed) or shown.
    """
    nsteps, nwalkers = nleaves.shape
    fig = plt.figure(figsize=(10, 6))
    for w in range(nwalkers):
        plt.plot(range(nsteps), nleaves[:, w], color=f"C{w % 10}", linewidth=1.5, alpha=0.8, rasterized=True)
    plt.xlabel('Sampler Iteration')
    plt.ylabel('Number of Leaves')
    plt.title('Evolution of Number of Leaves in Cold Chain')
    save_or_show(fig, filename)

def plot_acceptance_fraction(steps: typing.Union[np.ndarray, list],
                            total_acceptance_fraction: np.ndarray,
                             moves_acceptance_fraction: dict,
                             filename: str = None):
    """Plot the cold-chain acceptance fraction history, total and per move.

    Each curve is the walker mean of temperature 0; a dashed reference line marks 0.234.

    Args:
        steps: Sampler iterations of the recorded points, length ``nrec``.
        total_acceptance_fraction: Total acceptance fractions, shape
            ``(nrec, ntemps, nwalkers)``.
        moves_acceptance_fraction: ``{move name: array}``, each of shape
            ``(nrec, ntemps, nwalkers)``; may be empty.
        filename: Output path. Default ``None`` (show).

    Returns:
        None: The figure is saved (and closed) or shown.
    """

    fig = plt.figure(figsize=(10, 6))
    # cold chain total acceptance fraction
    plt.plot(steps, total_acceptance_fraction[:, 0].mean(axis=1), label='Total', color='black', linewidth=2)
    
    # skip if moves_acceptance_fraction is empty
    if len(moves_acceptance_fraction) != 0:
        for move, acc_fraction in moves_acceptance_fraction.items():
            plt.plot(steps, acc_fraction[:, 0].mean(axis=1), marker='o', label=move)

    plt.axhline(y=0.234, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='0.234')
    plt.legend()
    plt.xlabel('Sampler Iteration')
    plt.ylabel('Acceptance Fraction')  
    plt.title('Acceptance Fraction Over Time')

    save_or_show(fig, filename)

def plot_tempered_acceptance_fraction(steps: typing.Union[np.ndarray, list],
                            total_acceptance_fraction: np.ndarray,
                            palette: str = None,
                            filename: str = None):
    """Plot the total acceptance fraction history of every temperature.

    One curve per temperature (walker mean), with a colour-gradient legend from ``T_0`` to
    ``T_max``.

    Args:
        steps: Sampler iterations of the recorded points, length ``nrec``.
        total_acceptance_fraction: Total acceptance fractions, shape
            ``(nrec, ntemps, nwalkers)``.
        palette: Seaborn palette name or list of colours. Default ``None`` (``"icefire"``).
        filename: Output path. Default ``None`` (show).

    Returns:
        None: The figure is saved (and closed) or shown.
    """
    ntemps = total_acceptance_fraction.shape[1]
    tempcolors = sns.color_palette(palette if palette is not None else DEFAULT_PALETTE, ntemps)

    fig = plt.figure(figsize=(10, 6))
    
    for temp in range(ntemps):
        plt.plot(steps, total_acceptance_fraction[:, temp].mean(axis=1), color=tempcolors[temp], linewidth=1.5, marker='o', alpha=0.8, rasterized=True)

    ax = plt.gca()
    legend_width = 0.15
    legend_height = 0.03
    legend_x = 0.75
    legend_y = 0.9
    
    # Create gradient patches
    for i, color in enumerate(tempcolors[::-1]):  # Reverse to show cold->hot left to right
        rect = Rectangle(
            (legend_x + i * legend_width / ntemps, legend_y),
            legend_width / ntemps, legend_height,
            transform=ax.transAxes,
            facecolor=to_rgba(color, 0.7),
            edgecolor='none'
        )
        ax.add_patch(rect)
    
    # Add border and labels
    border = Rectangle(
        (legend_x, legend_y), legend_width, legend_height,
        transform=ax.transAxes,
        facecolor='none',
        edgecolor='black',
        linewidth=0.5
    )
    ax.add_patch(border)
    
    # Add text labels
    ax.text(legend_x - 0.01, legend_y + legend_height / 2, r'$T_{\rm max}$',
            transform=ax.transAxes, ha='right', va='center', fontsize=11, fontweight='normal', antialiased=True)
    ax.text(legend_x + legend_width + 0.01, legend_y + legend_height / 2, r'$T_0$',
            transform=ax.transAxes, ha='left', va='center', fontsize=11, fontweight='normal', antialiased=True)

    plt.legend()
    plt.xlabel('Sampler Iteration')
    plt.ylabel('Acceptance Fraction')  

    ymin, ymax = plt.ylim()
    plt.ylim(ymin, 1.2 * ymax)

    plt.title('Acceptance Fraction Over Time')

    save_or_show(fig, filename)

def plot_act_evolution(chain: dict,
                       iteration: int = 0,
                       parent_folder: str = '.'):
    
    """Plot how the cold-chain integrated autocorrelation time grows with chain length.

    For 10 log-spaced chain lengths from ``min(100, iteration)`` to ``iteration``, computes
    :func:`~eryn.utils.utility.get_integrated_act` of the cold chain (walker-averaged) and
    plots, per branch, the first ``ndim`` entries (the first leaf's parameters) against the
    ``tau = N / 50`` reference line, then the per-branch maximum in a combined figure.
    Branches with NaNs in the cold chain are skipped with a printed notice.

    Args:
        chain: ``{branch_name: chain of shape (nsteps, ntemps, nwalkers, nleaves, ndim)}``.
        iteration: Current iteration; sets the chain lengths and must be ``>= 1``. Default
            ``0``.
        parent_folder: Output folder: ``<parent_folder>/<branch>/act_evolution.png`` and
            ``<parent_folder>/max_act_evolution.png``. Default ``"."``.

    Returns:
        None: The figures are saved and closed.
    """

    NPOINTS = 10
    points = np.exp(np.linspace(np.log(min(100, iteration)), np.log(iteration), NPOINTS)).astype(int)
    
    taus = {}
    for branch, samples in chain.items():
        # create branch folder
        branch_folder = os.path.join(parent_folder, branch)
        os.makedirs(branch_folder, exist_ok=True)

        nsteps, ntemps, nwalkers, nleaves, ndim = samples.shape
        cold_chain = samples[:, 0, :, :, :].reshape(nsteps, 1, nwalkers, nleaves * ndim)
        #cold_chain = cold_chain[~np.isnan(cold_chain).any(axis=-1)] # remove NaNs
        if np.isnan(cold_chain).any(axis=-1).any():
            print(f"Skipping ACT plot for branch {branch} due to NaNs in the cold chain.")
            continue
        tmp = []
        for i, point in enumerate(points):
            if point > cold_chain.shape[0]:
                continue
            
            tau = get_integrated_act(cold_chain[:point], average=True)
            tmp.append(tau.squeeze())

        taus[branch] = np.array(tmp)
        
        fig = plt.figure(figsize=(10, 6))
        for d in range(ndim):
            plt.loglog(points[:len(taus[branch])], taus[branch][:, d], marker='o', label=fr'$x_{d}$')

        # add tau = iteration / 50 line for reference
        xaxis = np.logspace(0, np.log10(iteration), 100)
        #xaxis[0] = 1  # avoid plotting tau=0 at step 0

        plt.loglog(xaxis, xaxis / 50, linestyle='--', color='k', lw=2, label=r'$\tau$ = Nsteps / 50')
        plt.xlabel('Number of Steps')
        plt.ylabel('Integrated Autocorrelation Time')
        plt.title(f'Autocorrelation Time Evolution - Branch: {branch}')

        ymax = np.max(taus[branch]) * 1.2
        ymin = np.min(taus[branch]) * 0.5
        if ymin < ymax:
            plt.ylim(ymin, ymax)
        else:
            plt.ylim(0, ymax)
        plt.xlim(min(points) * 0.9, iteration * 1.1)
        plt.legend()
        save_or_show(fig, os.path.join(branch_folder, f'act_evolution.png'))
    
        # plot the maximum ACT across all parameters for each branch
    if len(taus) > 0:
        fig = plt.figure(figsize=(10, 6))
        for branch in taus.keys():
            max_tau = np.max(taus[branch], axis=1)
            plt.loglog(points[:len(max_tau)], max_tau, marker='o', label=f'Branch: {branch}')
        
        plt.loglog(xaxis, xaxis / 50, linestyle='--', color='k', lw=2, label=r'$\tau$ = Nsteps / 50')
        plt.xlabel('Number of Steps')
        plt.ylabel('Maximum Integrated Autocorrelation Time')
        plt.title('Maximum Autocorrelation Time Evolution Across Branches')
        ymax = max([np.max(np.max(taus[branch], axis=1)) for branch in taus.keys()]) * 1.2
        ymin = min([np.min(np.max(taus[branch], axis=1)) for branch in taus.keys()]) * 0.5
        if ymin < ymax:
            plt.ylim(ymin, ymax)
        else:
            plt.ylim(0, ymax)
        plt.xlim(min(points) * 0.9, iteration * 1.1)
        plt.legend()
        save_or_show(fig, os.path.join(parent_folder, f'max_act_evolution.png'))
            

def produce_base_plots(chain: dict, 
                       logl: np.ndarray, 
                       truths: dict = None, 
                       overlay_covariance: dict = None, 
                       labels: dict = None, 
                       iteration: int = 0,
                       parent_folder: str = '.',
                       ):

    """Write the standard diagnostic plots (the ``base`` group of :class:`PlotContainer`).

    For each branch (folder ``<parent_folder>/<branch>/``, created if needed):

    * ``cornerplot.png``: :func:`cornerplot` of the cold chain (all walkers and leaves,
      rows with NaNs dropped), with ``truths`` and the optional covariance overlay;
    * ``traceplot.png``: :func:`traceplot` of the cold chain.

    Plus ``<parent_folder>/loglikelihood.png`` (and ``loglikelihood_facet.png``) from
    :func:`plot_loglikelihood` on the cold-chain log-likelihood.

    Args:
        chain: ``{branch_name: chain of shape (nsteps, ntemps, nwalkers, nleaves, ndim)}``.
        logl: Log-likelihood, shape ``(nsteps, ntemps, nwalkers)``.
        truths: ``{branch_name: np.ndarray}`` of truth values (``(ndim,)`` or
            ``(ntruths, ndim)``), also used as the overlay centre. Default ``None``.
        overlay_covariance: ``{branch_name: covariance matrix}`` to overlay on the corner
            plots. Default ``None``.
        labels: ``{branch_name: list of parameter labels}``. Default ``None``.
        iteration: Step number shown in the corner-plot legend. Default ``0``.
        parent_folder: Output folder. Default ``"."``.

    Returns:
        None: All figures are saved and closed.
    """

    # create dictionary with a color shade for each branch
    branches = list(chain.keys())
    legend_label = 'Samples at step %d' % iteration

    palette = 'Blues'
    colors = sns.color_palette(palette, n_colors=len(branches))
    #colors = sns.color_palette("mako", n_colors=len(branches)+2)[2:]  # avoid very light colors
    branch_colors = {branch: colors[i] for i, branch in enumerate(branches)}

    for branch, samples in chain.items():
        branch_labels = labels.get(branch, None) if labels else None
        branch_truths = truths.get(branch, None) if truths else None
        branch_cov = overlay_covariance.get(branch, None) if overlay_covariance else None

        # create branch folder
        branch_folder = os.path.join(parent_folder, branch)
        os.makedirs(branch_folder, exist_ok=True)

        nsteps, ntemps, nwalkers, nleaves, ndim = samples.shape
        cold_chain = samples[:, 0, :, :, :].reshape(-1, ndim)
        cold_chain = cold_chain[~np.isnan(cold_chain).any(axis=1)] # remove NaNs

        cornerplot(
            cold_chain,
            means=branch_truths,
            overlay_covariance=branch_cov,
            legend_label=legend_label,
            truths=branch_truths,
            overlay_label='Information Matrix Covariance' if branch_cov is not None else None,
            labels=branch_labels,
            color=branch_colors[branch],
            filename=os.path.join(branch_folder, f'cornerplot.png')
        )

        traceplot(
            samples[:, 0, :, :, :],
            labels=branch_labels,
            truths=branch_truths,
            filename=os.path.join(branch_folder, f'traceplot.png')
        )

    plot_loglikelihood(
        logl[:, 0, :],
        filename=os.path.join(parent_folder, f'loglikelihood.png')
    )

def produce_tempering_plots(chain: dict, 
                            betas: np.ndarray,
                            logl: np.ndarray,
                            swap_acceptance_fraction: np.ndarray,
                            labels: dict = None, 
                            parent_folder: str = '.',
                            palette: str = None
                            ):
    """Write the tempering diagnostics (the ``tempering`` group of :class:`PlotContainer`).

    * ``<parent_folder>/<branch>/tempering_ridgeplot.png``: :func:`tempering_ridgeplot`
      per branch;
    * ``<parent_folder>/swap_acceptance.png``: :func:`plot_swap_acceptance`;
    * ``<parent_folder>/logl_betas.png``: :func:`plot_logl_betas` (with the
      stepping-stone evidence);
    * ``<parent_folder>/betas_evolution.png``: :func:`plot_betas_evolution`.

    Args:
        chain: ``{branch_name: chain of shape (nsteps, ntemps, nwalkers, nleaves, ndim)}``.
        betas: Inverse temperatures, shape ``(nsteps, ntemps)``.
        logl: Log-likelihood, shape ``(nsteps, ntemps, nwalkers)``.
        swap_acceptance_fraction: Adjacent-pair swap acceptance, shape ``(ntemps - 1,)``.
        labels: ``{branch_name: list of parameter labels}``. Default ``None``.
        parent_folder: Output folder. Default ``"."``.
        palette: Seaborn palette name or list of colours. Default ``None`` (``"icefire"``).

    Returns:
        None: All figures are saved and closed.
    """

    for branch, samples in chain.items():
        branch_labels = labels.get(branch, None) if labels else None

        # create branch folder
        branch_folder = os.path.join(parent_folder, branch)
        os.makedirs(branch_folder, exist_ok=True)

        tempering_ridgeplot(
            samples,
            labels=branch_labels,
            palette=palette,
            filename=os.path.join(branch_folder, f'tempering_ridgeplot.png')
        )

    plot_swap_acceptance(
        swap_acceptance_fraction,
        palette=palette,
        filename=os.path.join(parent_folder, f'swap_acceptance.png')
    )

    plot_logl_betas(
        betas,
        logl,
        palette=palette,
        filename=os.path.join(parent_folder, f'logl_betas.png')
    )

    plot_betas_evolution(
        betas,
        palette=palette,
        filename=os.path.join(parent_folder, f'betas_evolution.png')
    )

def produce_advanced_plots(steps: typing.Union[np.ndarray, list],
                           total_acceptance_fraction: np.ndarray,
                           moves_acceptance_fraction: dict,
                           palette: str = None,
                           iteration: int = 0,
                           chain: dict = None,
                           parent_folder: str = '.'):
    """Write the advanced diagnostics (the ``advanced`` group of :class:`PlotContainer`).

    * ``<parent_folder>/acceptance_fraction.png``: :func:`plot_acceptance_fraction`
      (cold chain, total and per move);
    * ``<parent_folder>/tempered_acceptance_fraction.png``:
      :func:`plot_tempered_acceptance_fraction` (total, per temperature);
    * the autocorrelation-time plots of :func:`plot_act_evolution`.

    Args:
        steps: Sampler iterations of the recorded acceptance points, length ``nrec``.
        total_acceptance_fraction: Shape ``(nrec, ntemps, nwalkers)``.
        moves_acceptance_fraction: ``{move name: array}``, each of shape
            ``(nrec, ntemps, nwalkers)``.
        palette: Seaborn palette name or list of colours. Default ``None`` (``"icefire"``).
        iteration: Current iteration, forwarded to :func:`plot_act_evolution`. Default
            ``0``.
        chain: ``{branch_name: chain of shape (nsteps, ntemps, nwalkers, nleaves, ndim)}``
            for the autocorrelation plots. Default ``None``.
        parent_folder: Output folder. Default ``"."``.

    Returns:
        None: All figures are saved and closed.
    """

    plot_acceptance_fraction(
        steps,
        total_acceptance_fraction,
        moves_acceptance_fraction,
        filename=os.path.join(parent_folder, f'acceptance_fraction.png')
    )

    plot_tempered_acceptance_fraction(
        steps,
        total_acceptance_fraction,
        palette=palette,
        filename=os.path.join(parent_folder, f'tempered_acceptance_fraction.png')
    )

    plot_act_evolution(
        chain,
        iteration=iteration,
        parent_folder=parent_folder
    )

def produce_rj_plots(nleaves: dict,
                     nleaves_min: dict,
                     nleaves_max: dict,
                     palette: str = None,
                     parent_folder: str = '.',
                     iteration: int = 0):
    
    """Write the reversible-jump diagnostics (the ``rj`` group of :class:`PlotContainer`).

    For each branch with ``nleaves_max > nleaves_min``, writes
    ``<parent_folder>/<branch>/rj_nleaves.png`` with :func:`plot_leaves`; other branches
    are skipped.

    Args:
        nleaves: ``{branch_name: leaf counts of shape (nsteps, ntemps, nwalkers)}``.
        nleaves_min: ``{branch_name: int}``; a missing branch counts as 1.
        nleaves_max: ``{branch_name: int}``; a missing branch counts as 1.
        palette: Seaborn palette name or list of colours. Default ``None`` (``"icefire"``).
        parent_folder: Output folder. Default ``"."``.
        iteration: Step number shown on the figures. Default ``0``.

    Returns:
        None: All figures are saved and closed.
    """

    for branch, leaves in nleaves.items():
        branch_nleaves_min = nleaves_min.get(branch, 1)
        branch_nleaves_max = nleaves_max.get(branch, 1)

        # skip if there is no reversible-jump sampling for this branch
        if branch_nleaves_max <= branch_nleaves_min:
            continue
        # create branch folder
        branch_folder = os.path.join(parent_folder, branch)
        os.makedirs(branch_folder, exist_ok=True)
        
        plot_leaves(
            leaves,
            branch_nleaves_min,
            branch_nleaves_max,
            palette=palette,
            iteration=iteration,
            filename=os.path.join(branch_folder, f'rj_nleaves.png')
        )

        # plot_leaves_evolution(
        #     leaves[:, 0],
        #     filename=os.path.join(branch_folder, f'rj_nleaves_evolution.png')
        # )
    
    


class PlotContainer:
    """Generate the runtime diagnostic plots from a backend.

    Not an :class:`~eryn.utils.updates.Update`: the sampler calls :meth:`produce_plots`
    directly every ``plot_iterations`` steps (and sets :attr:`backend` to its own backend
    when the container is passed as ``plot_generator``). Each call reads the chain,
    log-likelihood and ``betas`` from the backend and writes the selected plot groups to
    ``<parent_folder>/<group>/`` (see the module docstring for the groups).

    Construction calls :func:`setup_plotting` (global matplotlib style; prints a notice if
    LaTeX is missing) and creates ``parent_folder``.

    Args:
        backend: Backend to read (an :class:`~eryn.backends.Backend` or a reader with
            ``iteration``, ``key_order``, ``get_chain``, ``get_log_like``, ``get_betas``,
            and for the ``tempering`` / ``advanced`` / ``rj`` groups ``swaps_accepted``,
            ``nwalkers``, ``accepted``, ``rj``, ``get_nleaves``, ``nleaves_max``,
            ``rj_branches``). Default ``None`` (set it before plotting).
        plots: ``"base"``, ``"tempering"``, ``"advanced"``, ``"rj"``, ``"all"``, or a list
            of the first four. Default ``"base"``.
        branches: Branch names whose chains are plotted. Default ``None`` (all). The
            ``rj`` group is not filtered by this.
        truths: ``{branch_name: np.ndarray}`` of truth values for the ``base`` plots.
            Default ``None``.
        overlay_covariance: ``{branch_name: covariance matrix}`` overlaid on the corner
            plots. Default ``None``.
        tempering_palette: Seaborn palette name or list of colours for the per-temperature
            plots. Default ``None`` (``"icefire"``).
        parent_folder: Output folder (created). Default ``"."``.
        discard: Leading samples dropped before plotting: a count when ``>= 1``, else a
            fraction of ``backend.iteration``. Default ``0``.
        stop: :meth:`produce_plots` does nothing once ``backend.iteration > stop``. Default
            ``10000``.

    Attributes:
        plots (list[str]): Selected plot groups.
        steps (list[int]): Iterations at which the ``advanced`` group recorded acceptance.
        total_acceptance_fraction (np.ndarray or None): Recorded total acceptance
            fractions, shape ``(nrec, ntemps, nwalkers)``.
        move_acceptance_fractions (dict): Recorded per-move acceptance fractions.

    Raises:
        ValueError: If ``plots`` names an unknown group.

    Example:
        .. code-block:: python

            from eryn.ensemble import EnsembleSampler
            from eryn.utils import PlotContainer

            plotter = PlotContainer(
                plots=["base", "tempering"], parent_folder="diagnostics/", discard=0.3
            )
            sampler = EnsembleSampler(
                nwalkers, ndim, log_like_fn, priors,
                tempering_kwargs=dict(ntemps=4),  # the "tempering" group needs ntemps > 1
                plot_iterations=100, plot_generator=plotter,
            )
    """
    
    def __init__(self, 
                 backend: Backend = None,
                 plots : typing.Union[list, str] = 'base',
                 branches : list = None,
                 truths: dict = None,
                 overlay_covariance: dict = None,
                 tempering_palette: str = None,
                 parent_folder: str = '.',
                 discard: float = 0,
                 stop: int = int(1e4), 
                 ):
        """Initialize the PlotContainer (arguments are documented on the class)."""
        setup_plotting()
        self.backend = backend

        self.parent_folder = parent_folder
        os.makedirs(self.parent_folder, exist_ok=True)

        allowable_plots = ['base', 'tempering', 'advanced', 'rj']
        self.branches = branches

        if isinstance(plots, str):
            if plots == 'all':
                plots = allowable_plots
            else:
                plots = [plots]

        for plot in plots:
            if plot not in allowable_plots:
                raise ValueError(f"Plot type '{plot}' not recognized. Allowable types: {allowable_plots}")
        self.plots = plots
        
        #self.labels = labels
        self.truths = truths
        self.overlay_covariance = overlay_covariance
        self.discard = discard
        self.tempering_palette = tempering_palette

        self.steps = []
        self.total_acceptance_fraction = None
        self.move_acceptance_fractions = {}

        self.stop = stop

    @property
    def backend(self):
        """The backend (or reader) the plots are drawn from."""
        return self._backend
    @backend.setter
    def backend(self, value):
        self._backend = value

    @property
    def truths(self):
        """``{branch_name: truth values}`` for the ``base`` plots, or ``None``."""
        return self._truths
    @truths.setter
    def truths(self, value):
        self._truths = value

    @property
    def overlay_covariance(self):
        """``{branch_name: covariance matrix}`` overlaid on the corner plots, or ``None``."""
        return self._overlay_covariance
    @overlay_covariance.setter
    def overlay_covariance(self, value):
        self._overlay_covariance = value

    def produce_plots(self, sampler=None) -> None:
        """Read the backend and write every selected plot group.

        Does nothing once ``backend.iteration > stop``. Uses ``backend.key_order`` as the
        per-branch parameter labels. The ``tempering`` group uses the swap acceptance
        ``swaps_accepted / (iteration * nwalkers)``. The ``advanced`` group appends the
        current acceptance fractions to the recorded history before plotting and reads the
        full (undiscarded) chain for the autocorrelation plots. The ``rj`` group is skipped
        when ``backend.rj`` is ``False``.

        Args:
            sampler (EnsembleSampler, optional): The running sampler. Supplies the moves
                (per-move acceptance in ``advanced``) and ``nleaves_min`` (``rj``; else 0
                is used for every RJ branch). Default ``None``: plot from the backend alone.

        Returns:
            None: The figures are saved under ``parent_folder``.

        Raises:
            NotImplementedError: If the backend has ``nsamplers > 1``.

        Note:
            With ``sampler=None`` the ``advanced`` group raises ``UnboundLocalError`` (a
            ``NameError``): the fallback test ``hasattr(self.backend, moves)`` reads the
            not-yet-assigned local ``moves`` instead of the string ``"moves"``.
        """

        if getattr(self.backend, "nsamplers", 1) > 1:
            raise NotImplementedError(
                "PlotContainer does not support nsamplers > 1; plot per sampler "
                "with get_chain(sampler_index=...)."
            )

        if self.backend.iteration > self.stop:
            return

        labels = self.backend.key_order
    
        discard = int(self.discard) if self.discard >= 1 else int(self.discard * self.backend.iteration) 
        chain = self.backend.get_chain(discard=discard)
        logl = self.backend.get_log_like(discard=discard)
        betas = self.backend.get_betas(discard=discard)

        if self.branches is not None:
            chain = {branch: chain[branch] for branch in self.branches if branch in chain}

        for plot in self.plots:
            base_folder = os.path.join(self.parent_folder, plot)
            os.makedirs(base_folder, exist_ok=True)    

            if plot == 'base':           
                produce_base_plots(
                    chain=chain,
                    logl=logl,
                    truths=self.truths,
                    overlay_covariance=self.overlay_covariance,
                    iteration=self.backend.iteration,
                    labels=labels,
                    parent_folder=base_folder
                )
            elif plot == 'tempering':      
                swap_acceptance_fraction = self.backend.swaps_accepted / float(self.backend.iteration * self.backend.nwalkers)         
                produce_tempering_plots(
                    chain=chain,
                    betas=betas,
                    logl=logl,
                    swap_acceptance_fraction=swap_acceptance_fraction,
                    labels=labels,
                    parent_folder=base_folder,
                    palette=self.tempering_palette
                )

            elif plot == 'advanced':
                self.steps.append(self.backend.iteration)
                if self.total_acceptance_fraction is None:
                    self.total_acceptance_fraction = (self.backend.accepted / float(self.backend.iteration))[np.newaxis, ...]
                else:
                    self.total_acceptance_fraction = np.vstack((self.total_acceptance_fraction, (self.backend.accepted / float(self.backend.iteration))[np.newaxis, ...])) # shape (niterations, ntemps, nwalkers)
                
                if sampler is not None:
                    moves = sampler.moves
                elif hasattr(self.backend, moves):
                    moves = self.backend.moves
                else:
                    moves = None

                if moves is not None:
                    for move in moves:
                        name = move.__class__.__name__
                        if name not in self.move_acceptance_fractions:
                            self.move_acceptance_fractions[name] = move.acceptance_fraction[np.newaxis, ...]
                        else:
                            self.move_acceptance_fractions[name] = np.vstack((self.move_acceptance_fractions[name], move.acceptance_fraction[np.newaxis, ...])) # shape (niterations, ntemps, nwalkers)

                full_chain = self.backend.get_chain(discard=0) if discard > 0 else chain
                
                produce_advanced_plots(steps=self.steps,
                                        total_acceptance_fraction=self.total_acceptance_fraction,   
                                        moves_acceptance_fraction=self.move_acceptance_fractions,
                                        palette=self.tempering_palette,
                                        iteration=self.backend.iteration,
                                        chain=full_chain,
                                        parent_folder=base_folder
                                    )

            elif plot == 'rj':
                if self.backend.rj is False:
                    continue
    
                nleaves = self.backend.get_nleaves(discard=discard)

                nleaves_min = sampler.nleaves_min if sampler is not None else dict(zip(self.backend.rj_branches, [0]*len(self.backend.rj_branches)))
                nleaves_max = self.backend.nleaves_max

                produce_rj_plots(
                    nleaves=nleaves,
                    nleaves_min=nleaves_min,
                    nleaves_max=nleaves_max,
                    palette=self.tempering_palette,
                    parent_folder=base_folder,
                    iteration=self.backend.iteration
                )
