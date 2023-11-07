def plot_many(*imgs, title=None, titles=None, output_path=None, show=True, rows=None, **kwargs):
    """
    Plot multiple images in a row/grid.

    :param imgs: list of images to plot
    :param title: figure title
    :param titles: per-image titles
    :param output_path: save figure to this path if not None
    :param show: toggle showing the figure
    :param rows: number of rows in the grid (if None, automatically determined)
    :param kwargs: keyword arguments for imshow
    """
    from torch import Tensor as torch_tensor
    from matplotlib import pyplot as plt
    from PIL.Image import Image as PILImage
    import numpy as np

    if len(imgs) == 1 and isinstance(imgs[0], (list, tuple)):
        # unwrap imgs object if necessary (should be passed as plot_many(*imgs),
        # but sometimes I forget and pass plot_many(imgs))
        imgs = imgs[0]
    imgs = list(imgs)  # if tuple, convert to list
    total = len(imgs)
    if rows is None:  # determine number of rows automatically
        rows = 1 if total < 4 else int(np.ceil(np.sqrt(total)))
        cols = int(np.ceil(total / rows))
        rows, cols = min(rows, cols), max(rows, cols)
    else:
        cols = int(np.ceil(total / rows))

    # fill rectangle with white 2x2 images if necessary
    if total < rows * cols:
        imgs.extend([np.ones((2, 2, 3))] * (rows * cols - total))

    fig, ax = plt.subplots(rows, cols, figsize=(2.5 * cols, 3 * rows))
    fig.suptitle(title)
    for i, img in enumerate(imgs):
        # select current axis
        if total == 1:
            ax_i = ax
        elif rows == 1:
            ax_i = ax[i]
        else:
            ax_i = ax[i // cols, i % cols]  # indexing correct, read properly!

        if isinstance(img, torch_tensor):
            img = np.array(img.cpu())

        if isinstance(img, PILImage):
            img = np.array(img)
        if img.ndim == 4:
            img = img[0, ...]
        if img.shape[0] in [1, 3] and img.ndim == 3:
            img = img.transpose(1, 2, 0)

        ax_i.imshow(img, **kwargs)
        ax_i.axis('off')
        if titles is not None and i < len(titles):
            ax_i.set_title(titles[i])

    if rows == 2:
        # make more vertical space in between
        plt.subplots_adjust(hspace=0.3)

    plt.tight_layout(pad=0.8)
    if output_path is not None:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.0)
    if show:
        plt.show()
    plt.close()