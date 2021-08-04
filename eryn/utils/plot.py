import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl

mpl.rcParams[
    "text.usetex"
] = False  # TODO: Handle this properly. If left untreated it fails for people who use tex by default
import matplotlib.pyplot as plt
import corner.corner


class PlotContainer:
    """Automatic plotting and diagnostic information

    This class directs creation of plots. It can be used after MCMC
    runs to easily build plots and diagnostic information. It can also
    be used during runs for consistently updating diagnostic information
    about the current run.

    Args:
        fp (str, optional): File name for output pdf. (default: output)
        backend (object, optional): :class:`eryn.backends.Backend` object that
            holds MCMC data. (default: ``None``)
        thin_chain_by_ac(bool, optional): If True, thin the chain by half the minimum
            autocorrelation length and use a burnin of twice the max autocorrelation length.
            (default: ``False``)
        corner_kwargs (dict, optional): Keyword arguments for building corner
            plots. This can add extra key-value pairs or overwrite defaults.
            Defaults can be found with ``PlotContainer().default_corner_kwargs``.
        parameter_transforms (object, optional): :class:`eryn.utils.TransformContainer`
            object used to convert parameters to desired values for plotting.
            (default: ``None``)
        info_keys (list, optional): List of ``str`` indicating which keys from
            the information dictionary provided by the backend are of interest
            for diagnostics. (default: ``None``)

    """

    def __init__(
        self,
        fp="output",
        backend=None,
        thin_chain_by_ac=False,
        corner_kwargs={},
        parameter_transforms=None,
        info_keys=None,
    ):

        self.backend = backend
        self.fp = fp
        self.thin_chain_by_ac = thin_chain_by_ac

        if parameters_transforms is not None and not isinstance(
            parameter_transforms, TransformContainer
        ):
            raise ValueError(
                "If using parameter_transforms, must be eryn.utils.TransformContainer object."
            )
        self.parameter_transforms = parameter_transforms
        self.corner_kwargs = corner_kwargs

        self.injection = self.backend.truth
        if self.injection is not None and len(self.injection) == 0:
            self.injection = None

        for key, default in self.default_corner_kwargs.items():
            self.corner_kwargs[key] = self.corner_kwargs.get(key, default)

        self.info_keys = info_keys

    def transform(self, info):
        """Transform the samples in the infromation dictionary

        Args:
            info (dict): Information dictionary from the backend.

        """
        if self.parameter_transforms is not None:
            info["samples"] = self.parameter_transforms(info["samples"])
        return info

    @property
    def default_corner_kwargs(self):
        default_corner_kwargs = dict(
            levels=(1 - np.exp(-0.5 * np.array([1, 2, 3]) ** 2)),
            bins=25,
            plot_density=False,
            plot_datapoints=False,
            smooth=0.4,
            contour_kwargs={"colors": "blue"},
            hist_kwargs={"density": True},
            truths=self.injection,
            show_titles=True,
            title_fmt=".2e",
        )
        return default_corner_kwargs

    @property
    def info_keys(self):
        return self._info_keys

    @info_keys.setter
    def info_keys(self, info_keys):
        if info_keys is not None:
            if not isinstance(info_keys, list):
                raise ValueError("info_keys must be a list.")

            self._info_keys = info_keys

        else:
            self._info_keys = [
                "ntemps",
                "nwalkers",
                "nbranches",
                "max logl",
                "shapes",
            ]

    def add_backend(self, backend, custom_backend=False):
        """Add a backend after initialization

        Args:
            backend (object): Either a :class:`eryn.backends.Backend`
                or :class:`eryn.backends.HDFBackend` object or a custom backend
                object.
            custom_backend (bool, optional): If using a custom backend class,
                this should be True. (default: ``False``)

        """
        # custom_backend is if they make their own
        if (
            not isinstance(backend, Backend) and not isinstance(backend, HDFBackend)
        ) and not custom_backend:
            raise ValueError("Backend must be a default backend")

        self.backend = backend

    def generate_corner(
        self, burn=0, thin=1, pdf=None, name=None, info=None, **corner_kwargs
    ):
        """Build a corner plot

        This function builds a corner plot to be added to a pdf.

        Args:
            burn (int, optional): Number of samples to burn. If
                ``self.thin_chain_by_ac == True``, then this is overridden.
                (default: ``0``)
            thin (int, optional): Number of samples to burn. If
                ``self.thin_chain_by_ac == True``, then this is overridden.
                (default: ``1``)
            pdf (object, optional): An open PdfPages object
                (`see her for an example <https://matplotlib.org/stable/gallery/misc/multipage_pdf.html>`_).
                It will not be closed by this function. If not provided, a new pdf
                will be opened, added to, then closed.
                (default: ``None``)
            name (str, optional): If not providing ``pdf`` kwarg, ``name`` will be
                the name of the pdf document that is created and saved.
                (default: ``None``)
            info (dict, optional): Information dictionary from the backend. If not
                provided, it will be retrieved from the backend.
                (default: ``None``)
            corner_kwargs (dict, optional): Pass kwarg arguments direct to
                the corner plot. This will temperorarily overwrite entries in
                the ``self.corner_kwargs`` attribute.


        """

        # get info from backend
        if info is None and self.backend is not None:
            info = self.transform(self.backend.get_info(discard=burn, thin=thin))

        else:
            raise ValueError("Need to provide either info or self.backend.")

        if self.thin_chain_by_ac:
            burn = info["ac_burn"]
            thin = info["ac_thin"]

        # build corner_kwargs with self.corner_kwargs
        for key, val in self.corner_kwargs.items():
            corner_kwargs[key] = corner_kwargs.get(key, val)

        # adjust color info
        if "hist_kwargs" in corner_kwargs:
            if "color" in corner_kwargs["hist_kwargs"] and "color" in corner_kwargs:
                corner_kwargs["hist_kwargs"]["color"] = corner_kwargs["color"]

        # open new pdf if not provided
        if pdf is None:
            close_file = True
            name = self.fp if name is None else name
            pdf = PdfPages(name + ".pdf")
        else:
            close_file = False

        # make corner plot for each leaf
        # TODO: make corner plot across leaves
        for key, coords in info["samples"].items():
            nsteps, ntemps, nwalkers, nleaves_max, ndim = coords.shape
            for temp in range(ntemps):
                for leaf in range(nleaves_max):
                    # get samples
                    samples_in = coords[:, temp, :, leaf].reshape(-1, ndim)

                    # build corner figure
                    fig = corner.corner(samples_in, **corner_kwargs,)

                    # add informational title
                    fig.suptitle(
                        f"Branch: {key}\nTemperature: {temp}\nLeaf: {leaf}\nSample Size: {samples_in.shape[0]}"
                    )
                    # save to open pdf
                    pdf.savefig(fig)
                    # close the plot not the pdf
                    plt.close()

        # if pdf was created here, close it
        if close_file:
            pdf.close()

    def generate_info_page(self, burn=0, thin=1, pdf=None, name=None, info=None):
        """Build an info page

        This function puts an info page in a pdf.

        Args:
            burn (int, optional): Number of samples to burn. If
                ``self.thin_chain_by_ac == True``, then this is overridden.
                (default: ``0``)
            thin (int, optional): Number of samples to burn. If
                ``self.thin_chain_by_ac == True``, then this is overridden.
                (default: ``1``)
            pdf (object, optional): An open PdfPages object
                (`see her for an example <https://matplotlib.org/stable/gallery/misc/multipage_pdf.html>`_).
                It will not be closed by this function. If not provided, a new pdf
                will be opened, added to, then closed.
                (default: ``None``)
            name (str, optional): If not providing ``pdf`` kwarg, ``name`` will be
                the name of the pdf document that is created and saved.
                (default: ``None``)
            info (dict, optional): Information dictionary from the backend. If not
                provided, it will be retrieved from the backend.
                (default: ``None``)

        """
        # get information from backend
        if info is None and self.backend is not None:
            info = self.transform(self.backend.get_info(discard=burn, thin=thin))

        else:
            raise ValueError("Need to provide either info or self.backend.")

        # build info from long string
        title_str = self.fp + " informat:\n"

        for key in self.info_keys:

            if key not in ["shapes", "max logl"]:
                title_str += f"{key}: {info[key]}\n"

            elif key == "max logl":
                title_str += f"{key}: {info['log_prob'].max()}\n"

            elif key == "shapes":
                for key, shape in info["shapes"].items():
                    title_str += f"{key}:\n"
                    title_str += f"    shape: {shape}\n"
                    title_str += f"    nleaves max: {shape[2]}\n"
                    title_str += f"    ndim: {shape[3]}\n"

        fig = plt.Figure()
        fig.suptitle(title_str, fontsize=16, ha="left", x=0.25)

        # open pdf if not given
        if pdf is None:
            close_file = True
            name = self.fp if name is None else name
            pdf = PdfPages(name + ".pdf")
        else:
            close_file = False

        pdf.savefig(fig)

        plt.close()
        # close file if created here
        if close_file:
            pdf.close()

    def generate_plot_info(self, burn=0, thin=1, pdf=None, name=None, info=None):
        """Build an info page

        This function puts an info page in a pdf.

        Args:
            burn (int, optional): Number of samples to burn. If
                ``self.thin_chain_by_ac == True``, then this is overridden.
                (default: ``0``)
            thin (int, optional): Number of samples to burn. If
                ``self.thin_chain_by_ac == True``, then this is overridden.
                (default: ``1``)
            pdf (object, optional): An open PdfPages object
                (`see her for an example <https://matplotlib.org/stable/gallery/misc/multipage_pdf.html>`_).
                It will not be closed by this function. If not provided, a new pdf
                will be opened, added to, then closed.
                (default: ``None``)
            name (str, optional): If not providing ``pdf`` kwarg, ``name`` will be
                the name of the pdf document that is created and saved.
                (default: ``None``)
            info (dict, optional): Information dictionary from the backend. If not
                provided, it will be retrieved from the backend.
                (default: ``None``)

        """
        # must have backend in this case
        if info is None:
            info = self.transform(self.backend.get_info(discard=burn, thin=thin))

        if pdf is None:
            close_file = True
            name = self.fp if name is None else name
            pdf = PdfPages(name + ".pdf")
        else:
            close_file = False

        self.generate_info_page(info=info, pdf=pdf)
        self.generate_corner(info=info, pdf=pdf)

        # close file if created here
        if close_file:
            pdf.close()


if __name__ == "__main__":
    plot = PlotContainer("../GPU4GW/MBH_for_corner.h5", "mbh")

    plot.generate_corner()
