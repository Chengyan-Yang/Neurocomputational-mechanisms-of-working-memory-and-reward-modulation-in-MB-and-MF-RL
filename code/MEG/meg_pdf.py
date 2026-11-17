"""Publication PDF export for MEG plots."""
from pathlib import Path

import matplotlib as mpl
import numpy as np


def save_pdf(fig, path, **kwargs):
    """Keep text, heat-map cells and colour scales editable in the PDF."""
    for axis in fig.axes:
        for image in list(axis.images):
            values = image.get_array()
            if values.ndim != 2 or image.get_interpolation() not in {"nearest", "none"}:
                continue
            left, right, bottom, top = image.get_extent()
            if image.origin == "upper":
                bottom, top = top, bottom
            ny, nx = values.shape
            x = np.linspace(left, right, nx + 1)
            y = np.linspace(bottom, top, ny + 1)
            xlim, ylim = axis.get_xlim(), axis.get_ylim()
            mesh = axis.pcolormesh(x, y, values, shading="flat", cmap=image.cmap,
                                   norm=image.norm, alpha=image.get_alpha(),
                                   transform=image.get_transform(), zorder=image.get_zorder(),
                                   edgecolors="none", rasterized=False)
            mesh.set_clip_path(image.get_clip_path())
            mesh.set_clip_box(image.get_clip_box())
            mesh.set_clip_on(image.get_clip_on())
            image.remove()
            axis.set_xlim(xlim)
            axis.set_ylim(ylim)
        for collection in axis.collections:
            collection.set_rasterized(False)
    with mpl.rc_context({"pdf.fonttype": 42, "pdf.use14corefonts": False}):
        fig.savefig(Path(path), format="pdf", dpi=600, **kwargs)

def save_brain_png(brain, path, *, base_alpha=1.0):
    """Export the native 3-D view as one tightly cropped transparent image."""
    from PIL import Image, ImageOps

    rgba = brain._renderer.plotter.screenshot(
        return_img=True, transparent_background=True, scale=2
    )
    if base_alpha < 1.0 and rgba.shape[-1] == 4:
        # Apply the reference RSA cortex opacity before cropping the native view.
        rgba = rgba.copy()
        rgb = rgba[:, :, :3]
        soften = (rgb.max(axis=2) > 130) & (np.ptp(rgb, axis=2) < 45)
        soften[int(len(soften) * 0.83):] = False
        alpha = rgba[:, :, 3]
        alpha[soften] = np.minimum(alpha[soften], round(255 * base_alpha))
    image = Image.fromarray(rgba)
    if image.mode != "RGBA":
        raise RuntimeError("The 3-D renderer did not supply an alpha channel.")
    alpha = image.getchannel("A")
    bounds = alpha.getbbox()
    if bounds is None or alpha.getextrema()[0] != 0:
        raise RuntimeError("Expected a visible brain on a transparent background.")
    cropped = ImageOps.expand(image.crop(bounds), border=2, fill=(0, 0, 0, 0))
    cropped.save(path, format="PNG", dpi=(600, 600))


def pdf_size_inches(path):
    """Return the physical size of a generated single-page PDF."""
    from PIL.PdfParser import PdfParser

    with PdfParser(filename=str(path), mode="rb") as pdf:
        page = pdf.read_indirect(pdf.pages[0])
        left, bottom, right, top = page[b"MediaBox"]
        return (right - left) / 72, (top - bottom) / 72


def render_pdf_for_comparison(path, dpi):
    """Render an entire page for the optional RDM label-pixel audit."""
    import subprocess
    from tempfile import TemporaryDirectory
    from PIL import Image

    with TemporaryDirectory() as folder:
        prefix = Path(folder) / "page"
        subprocess.run(["pdftoppm", "-f", "1", "-l", "1", "-singlefile",
                        "-r", str(dpi), "-png", str(path), str(prefix)], check=True)
        with Image.open(prefix.with_suffix(".png")) as image:
            return np.array(image.convert("RGB"))
