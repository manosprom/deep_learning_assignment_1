def history(histories, metrics=["loss", "sparse_categorical_accuracy"], figsize=(20, 10)):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 12})
    len_metrics = len(metrics)
    # print(len_metrics)
    fig, ax = plt.subplots(1, len_metrics, figsize=figsize)


    for i, metric in enumerate(metrics):

        max_epochs = []
        for label in histories:
            ax[i].plot(histories[label].history[metric], label='{0:s} train {1:s}'.format(label, metric), linewidth=2)
            ax[i].plot(histories[label].history['val_{0:s}'.format(metric)], label='{0:s} validation {1:s}'.format(label, metric), linewidth=2)
            max_epochs.append(len(histories[label].history[metric]))
        ax[i].set_xticks(np.arange(0, max(max_epochs) + 10, 10))
        ax[i].legend()
