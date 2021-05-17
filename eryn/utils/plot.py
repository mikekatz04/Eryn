import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import corner.corner


class PlotContainer:
    def __init__(
        self,
        fp="output",
        backend=None,
        thin_chain_by_ac=False,
        corner_kwargs={},
        # test_inds=None,
        parameter_transforms=None,
        info_keys=None,
    ):

        if parameter_transforms is not None:
            # TODO:
            raise NotImplementedError

        self.backend = backend
        self.fp = fp
        self.thin_chain_by_ac = thin_chain_by_ac

        # TODO: deal with test_inds
        # self.test_inds = self.backend.get_attr("test_inds")

        self.parameter_transforms = parameter_transforms
        self.corner_kwargs = corner_kwargs

        self.injection = self.backend.truth

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

        for key, default in default_corner_kwargs.items():
            self.corner_kwargs[key] = self.corner_kwargs.get(key, default)

        self.info_keys = info_keys

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
        # custom_backend is if they make their own
        if (
            not isinstance(backend, Backend) and not isinstance(backend, HDFBackend)
        ) and not custom_backend:
            raise ValueError("Backend must be a default backend")

        self.backend = backend

    def generate_corner(
        self, burn=0, thin=1, pdf=None, name=None, info=None, **corner_kwargs
    ):

        if self.thin_chain_by_ac:
            burn = 0
            thin = 1

        if info is None and self.backend is not None:
            info = self.backend.get_info(burn=burn, thin=thin)

        if self.thin_chain_by_ac:
            burn = info["ac_burn"]
            thin = info["ac_thin"]

        # TODO: add valueerrors

        for key, val in self.corner_kwargs.items():
            corner_kwargs[key] = corner_kwargs.get(key, val)

        if "hist_kwargs" in corner_kwargs:
            if "color" in corner_kwargs["hist_kwargs"] and "color" in corner_kwargs:
                corner_kwargs["hist_kwargs"]["color"] = corner_kwargs["color"]

        if pdf is None:
            close_file = True
            name = self.fp if name is None else name
            pdf = PdfPages(name + ".pdf")
        else:
            close_file = False

        #  with PdfPages("temp_check" + ".pdf") as pdf:
        for name, coords in info["samples"].items():
            nsteps, ntemps, nwalkers, nleaves_max, ndim = coords.shape
            for temp in range(ntemps):
                for leaf in range(nleaves_max):
                    samples_in = coords[:, temp, :, leaf].reshape(-1, ndim)
                    fig = corner.corner(samples_in, **corner_kwargs,)
                    fig.suptitle(
                        f"Branch: {name}\nTemperature: {temp}\nLeaf: {leaf}\nSample Size: {samples_in.shape[0]}"
                    )
                    pdf.savefig(fig)
                    plt.close()

        if close_file:
            pdf.close()

    def generate_info_page(self, info=None, pdf=None, burn=0, thin=1):

        if info is None and self.backend is not None:
            info = self.backend.get_info(burn=burn, thin=thin)

        title_str = self.fp + " informat:\n"

        for key in self.info_keys:

            if key not in ["shapes", "max logl"]:
                title_str += f"{key}: {info[key]}\n"

            elif key == "max logl":
                title_str += f"{key}: {info['log_prob'].max()}\n"

            elif key == "shapes":
                for name, shape in info["shapes"].items():
                    title_str += f"{name}:\n"
                    title_str += f"    shape: {shape}\n"
                    title_str += f"    nleaves max: {shape[2]}\n"
                    title_str += f"    ndim: {shape[3]}\n"

        fig = plt.Figure()
        fig.suptitle(title_str, fontsize=16, ha="left", x=0.25)

        if pdf is None:
            close_file = True
            name = self.fp if name is None else name
            pdf = PdfPages(name + ".pdf")
        else:
            close_file = False

        pdf.savefig(fig)

        plt.close()
        if close_file:
            pdf.close()

    def generate_update(self, burn=0, thin=1, save=True, name=None, **kwargs):
        if self.backend is None:
            raise ValueError("Must initialize with a backend.")

        info = self.backend.get_info(burn=burn, thin=thin)

        name = self.fp if name is None else name
        with PdfPages(name + ".pdf") as pdf:
            self.generate_info_page(info=info, pdf=pdf)
            self.generate_corner(info=info, pdf=pdf)


if __name__ == "__main__":
    plot = PlotContainer("../GPU4GW/MBH_for_corner.h5", "mbh")

    plot.generate_corner()
