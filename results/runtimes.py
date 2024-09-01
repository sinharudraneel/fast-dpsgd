import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 200)
names    = ['PyTorch', 'Opacus',]
private  = [0, 1]
filenames= ['pytorch', 'opacusdp',]
expts = ['logreg', 'ffnn', 'mnist', 'embed', 'cifar10']
batch_sizes = [16, 32, 64, 128, 256]


assert len(names) == len(private) == len(filenames)

def expt_iterator():
    for expt in expts:
        for bs in batch_sizes:
            for dpsgd, name, filename in zip(private, names, filenames):
                yield expt, bs, name, filename, bool(dpsgd)

files = []
success, errors = 0, 0
for expt, bs, name, filename, dpsgd in expt_iterator():
    pickle_name = f'./raw/{filename}_{expt}_bs_{bs}_priv_{dpsgd}'
    
    use_xla = 'xla' in name.lower() or name.lower().startswith('jax')
    if filename.startswith('tf'):
        pickle_name += f'_xla_{use_xla}'
    
    try:
        with open(pickle_name+'.pkl', 'rb') as f:
            d = pickle.load(f)
            success += 1
    except:
        print(f'Failed to load {pickle_name}.pkl')
        d = None
        errors += 1
    files.append((filename, name, expt, bs, dpsgd, use_xla, d))



success, errors

df_list = []
for *row, d in files:
    d = [np.median(d['timings'])] if d else [0.]
    df_list.append(pd.Series(row + d))

df = pd.concat(df_list, axis=1).transpose()
df.columns = ['Filename', 'Library', 'Experiment', 'Batch Size', 'Private?', 'XLA', 'Median Runtime']
df['Median Runtime'] = df['Median Runtime'].astype(float)

expt_to_title = {
    'mnist': 'MNIST Convolutional Neural Network',
    'embed': 'Embedding Network',
    'ffnn': 'Fully Connected Neural Network (FCNN)',
    'logreg': 'Logistic Regression',
    'cifar10': 'CIFAR10 Convolutional Neural Network'
}

def get_runtime_plot(expt, ylim=None, figsize=(13, 6)):
    f, ax = plt.subplots(2, 1, figsize=figsize, sharey=True)
    plot_df = df[df['Experiment'] == expt].copy()
    if ylim:
        plot_df['Median Runtime'] = np.minimum(plot_df['Median Runtime'], ylim-2)

    sns.barplot(x='Library', y='Median Runtime', hue='Batch Size', data=plot_df[plot_df['Private?']], ax=ax[0], palette='muted')
    sns.barplot(x='Library', y='Median Runtime', hue='Batch Size', data=plot_df[plot_df['Private?'] != True], ax=ax[1], palette='muted')

    for ax_ind, private in enumerate([True, False]):
        tmp = df.loc[(df['Experiment'] == expt) & (df['Private?'] == private), 'Median Runtime']
        for i, (rect, tim) in enumerate(zip(ax[ax_ind].patches, tmp)):
            height = rect.get_height()
            if tim > 100.:
                annotation = f'{int(tim)}'
            elif tim > 0.:
                annotation = f'{tim:.2g}'
            else:
                annotation = ''
            ax[ax_ind].annotate(annotation,
                                xy=(rect.get_x() + rect.get_width() / 2 - 0.3*rect.get_width(), height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                va='bottom', ha='left', 
                                fontsize=9, rotation=45)



    plt.title('')
    if expt == 'cifar10':
        y = 1.15
    else:
        y = 1
    ax[0].set_title('Median Runtime for One "Private" Epoch (without clip/noise) - '+ expt_to_title[expt], 
                    y=y)
    ax[1].set_title('Median Runtime for One Non-Private Epoch - '+ expt_to_title[expt])
    ax[0].set_xlabel('Library')
    ax[1].set_xlabel('Library')
    ax[0].set_ylabel('Median Runtime (sec)')
    ax[1].set_ylabel('Median Runtime (sec)')
    if ylim:
        ax[0].set_ylim(0, ylim)
        ax[1].set_ylim(0, ylim)
    # ax[1].set_ylabel('')
    ax[0].get_legend().remove()
    ax[1].get_legend().remove()
    sns.despine()
    plt.legend()
    f.patch.set_facecolor('white')
    f.tight_layout()
    return f, ax

f, ax = get_runtime_plot('logreg', ylim=20, figsize=(11, 5))
f.savefig('logistic_runtimes_clip.pdf')
f, ax = get_runtime_plot('ffnn', 20, figsize=(11, 5))
f.savefig('ffnn_runtimes_clip.pdf')
f, ax = get_runtime_plot('mnist', 50, figsize=(11, 5))
f.savefig('cnn_runtimes_clip.pdf')
f, ax = get_runtime_plot('cifar10', 175, figsize=(11, 5))
f.savefig('cifar10_cnn_runtimes_clip.pdf')
f, ax = get_runtime_plot('embed', 20, figsize=(11, 5))
f.savefig('embed_runtimes_clip.pdf')