import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from bokeh.plotting import ColumnDataSource, figure
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from io import BytesIO
import base64

def get_svg_str(smiles, svg_size=200):
    d2d = Draw.rdMolDraw2D.MolDraw2DSVG(svg_size*2, svg_size)
    opts = d2d.drawOptions()
    opts.clearBackground = False
    d2d.DrawMolecule(Chem.MolFromSmiles(smiles))
    d2d.FinishDrawing()
    img_str = d2d.GetDrawingText()
    buffered = BytesIO()
    buffered.write(str.encode(img_str))
    img_str = base64.b64encode(buffered.getvalue())
    img_str = "data:image/svg+xml;base64,{}".format(repr(img_str)[2:-1])
    return img_str

def plot_with_hover(pcas_df, ifs, df_in, exp_var, imgs):

    print(len(list(set(imgs))))
    imgs = [get_svg_str(img) for img in imgs]
    source = ColumnDataSource(data=dict(x=pcas_df["PC1"], y=pcas_df["PC2"], z=pcas_df["PC3"], imgs=imgs))
    TOOLTIPS = """<img
                   src="@imgs" height="200" alt="@imgs" width="400"
                   style="float: left; margin: 0px 15px 15px 0px;"
                   border="2"
            ></img>"""
    p = figure(width=800, height=400, title=r"delta G v.s. delta H", tooltips=TOOLTIPS,
               x_axis_label=r"$$\Delta H (kcal/mol)$$", y_axis_label=r"$$\Delta G(kcal/mol)$$")
    p.scatter(x="x", y="y", source=source)
    return p

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

def min_max_scale(df, scaler=None):

    if scaler is None:
        scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    return df_scaled, scaler

def std_scale(df, scaler=None):

    if scaler is None:
        scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    return df_scaled, scaler

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

def run_PCA(in_array, cols, n_components, pca=None):

    if pca is not None:
        pca = PCA(n_components=n_components)
        pcas = pca.fit_transform(in_array)
        pcas_df = pd.DataFrame(pcas, columns=["PC"+f"{i+1}" for i in range(n_components)])
        return pcas_df
    else:
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
        return pcas_df, exp_var, ifs, pca

def pca_3d(pcas_df, ifs, df_in, exp_var, label="PC1", smiles_df=None, color_by_dataset=None):

    if color_by_dataset is not None:
        color_label = color_by_dataset
        fig = go.Figure(data=[go.Scatter3d(
        x=pcas_df["PC1"],
        y=pcas_df["PC2"],
        z=pcas_df["PC3"],
        mode='markers',
        name="original",
        marker=dict(
        color="blue",
        size=12        
        ))])
        fig.add_trace(go.Scatter3d(
        x=color_by_dataset["PC1"],
        y=color_by_dataset["PC2"],
        z=color_by_dataset["PC3"],
        mode='markers',
        name="STONED",
        marker=dict(
        color="red",
        size=12        
        )))
        #x_lbl = f"PC1"
        #y_lbl = f"PC2"
        #z_lbl = f"PC3"
        x_lbl = ""
        y_lbl = ""
        z_lbl = ""
        # tight layout
        fig.update_layout(title={"text":f"Principal Component Analysis Explained Variance  = {round(np.sum(exp_var), 2)*100}%"},
                        scene=dict(
                                    xaxis = dict(
                                        backgroundcolor="rgb(230, 230, 200)",
                                        gridcolor="white",
                                        showbackground=True,
                                        zerolinecolor="white",),
                                    yaxis = dict(
                                        backgroundcolor="rgb(230, 230, 200)",
                                        gridcolor="white",
                                        showbackground=True,
                                        zerolinecolor="white"),
                                    zaxis = dict(
                                        backgroundcolor="rgb(230, 230, 200)",
                                        gridcolor="white",
                                        showbackground=True,
                                        zerolinecolor="white",),
                                    xaxis_title=x_lbl,
                                    yaxis_title=y_lbl,
                                    zaxis_title=z_lbl
                                        ),
                        width=500,
                        height=700,
                        margin=dict(l=10, r=10, b=10, t=30),
                        legend=dict(groupclick="toggleitem"))
    else:
        color_label = list(ifs[label].keys())[0]
        #if smiles_df is not None:
        #    hover_data = smiles_df["SMILES"]
        #else:
        #    hover_data = None
        fig = go.Figure(data=[go.Scatter3d(
        x=pcas_df["PC1"],
        y=pcas_df["PC2"],
        z=pcas_df["PC3"],
        mode='markers',
        marker=dict(
            size=12,
            color=df_in[color_label],
            colorscale='solar',
            opacity=0.8, colorbar=dict(lenmode='fraction', len=0.75, thickness=20, title=f"{', '.join(list(ifs[label].keys()))}")
        ))])
        pc1 = [e + f' <b>IF #{i+1}</b><br>' for i, e in enumerate(list(ifs['PC1'].keys()))]
        pc2 = [e + f' <b>IF #{i+1}</b><br>' for i, e in enumerate(list(ifs['PC2'].keys()))]
        pc3 = [e + f' <b>IF #{i+1}</b><br>' for i, e in enumerate(list(ifs['PC3'].keys()))]
        x_lbl = f"PC1<br>{''.join(pc1)}"
        #x_lbl = ""
        #y_lbl = ""
        #z_lbl = ""
        y_lbl = f"PC2<br>{''.join(pc2)}"
        z_lbl = f"PC3<br>{''.join(pc3)}"
        # tight layout

        fig.update_layout(title={"text":f"<b>Principal Component Analysis Explained Variance  = {round(np.sum(exp_var), 2)*100}%</b>", 'xanchor': 'center', 'yanchor': 'top', 
                                 "font":dict(
                                            size=32,
                                            color='#000000'
                                        )},
                          scene=dict(
                                    xaxis = dict(
                                        backgroundcolor="rgb(230, 230, 200)",
                                        gridcolor="white",
                                        showbackground=True,
                                        zerolinecolor="white",),
                                    yaxis = dict(
                                        backgroundcolor="rgb(230, 230, 200)",
                                        gridcolor="white",
                                        showbackground=True,
                                        zerolinecolor="white"),
                                    zaxis = dict(
                                        backgroundcolor="rgb(230, 230, 200)",
                                        gridcolor="white",
                                        showbackground=True,
                                        zerolinecolor="white",),
                                    xaxis_title=x_lbl,
                                    yaxis_title=y_lbl,
                                    zaxis_title=z_lbl
                                        ),
                          width=500,
                          height=700,
                          margin=dict(l=40, r=40, b=10, t=40),
                          font=dict(family="helvetica", size=16)
                        )

    return fig