import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

def gen_single_figure(size=(20, 9), nrows=2, ncols=7):

    fig, axs = plt.subplots(figsize=size, nrows=nrows, ncols=ncols)
    for ax in axs.reshape(-1):
        ax.grid(True, linewidth=1.0, color='0.95')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_axisbelow(True)
        for axis in ['bottom', 'left']:
            ax.spines[axis].set_linewidth(2.5)
        for tick in ax.get_yticklabels():
            tick.set_fontname("Arial")
            tick.set_fontsize(12)
        for tick in ax.get_xticklabels():
            tick.set_fontname("Arial")
            tick.set_fontsize(12)
    
    return fig, axs

def load_descriptors(in_file):

    descriptors = pd.read_csv(in_file)
    return descriptors, descriptors.columns

def min_max_scale(df):

    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    return df_scaled

def std_scale(df):

    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    return df_scaled

def histogram(df, size=(20, 9), bins=20, nrows=2, ncols=7):

    fig, axs = gen_single_figure()
    cols = df.columns
    for i, ax in enumerate(axs.reshape(-1)):
        try:
            d = cols[i]
            data = df[d].tolist()
            ax.hist(data, color="lightsalmon", bins=bins)
            ax.set_ylabel("Frequency of Descriptor Bin", fontsize=10)
            ax.set_xlabel(f"Scaled Descriptor Value\n`{cols[i]}`", fontsize=10)
            ax.set_title(f"Distribution of Scaled\n`{cols[i]}` Values", fontsize=10)
        except:
            pass
    return fig, axs

def run_PCA(in_array, cols, n_components):
    pca = PCA(n_components=n_components)
    pcas = pca.fit_transform(in_array)
    pcas_df = pd.DataFrame(pcas, columns=["PC"+f"{i+1}" for i in range(n_components)])
    exp_var = pca.explained_variance_ratio_
    ifs = {"PC"+f"{i+1}": [] for i in range(n_components)}
    for i in range(n_components):
        pca_components = np.abs(pca.components_[i,:])
        indices = pca_components.argsort()
        descriptors = np.flip(np.asarray(cols)[indices])
        sorted_components = pca_components[indices]
        sum_components = np.sum(sorted_components)
        normalized_components = sorted_components / sum_components
        flipped = np.flip(normalized_components)
        d = {}
        for index, c in enumerate(flipped[:2]):
            d[descriptors[index]] = str(round(c, 2)*100) + "%"
        ifs["PC"+f"{i+1}"] = d

    return pcas_df, exp_var, ifs

def pca_3d(pcas_df, ifs, df_in, exp_var, label="PC1"):

    color_label = list(ifs[label].keys())[0]
    fig = go.Figure(data=[go.Scatter3d(
    x=pcas_df["PC1"],
    y=pcas_df["PC2"],
    z=pcas_df["PC3"],
    mode='markers',
    marker=dict(
        size=12,
        color=df_in[color_label],
        colorscale='solar',
        opacity=0.8, colorbar=dict(lenmode='fraction', len=0.75, thickness=20)
    ))])
    x_lbl = f"PC1"
    y_lbl = f"PC2"
    z_lbl = f"PC3"
    # tight layout
    fig.update_layout(title={"text":f"Principal Component Analysis<br>Explained Variance  = {round(np.sum(exp_var), 2)*100}%"},
                    scene=dict(xaxis_title = x_lbl,
                    yaxis_title = y_lbl,
                    zaxis_title = z_lbl),
                    width=700,
                    margin=dict(l=20, r=20, b=20, t=40),
                    )
    return fig
