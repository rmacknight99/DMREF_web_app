from app_utils import *
import streamlit as st
from desc_utils import *
from rdkit import Chem
import sys

def input_smiles():

    input_ = st.text_input("Enter SMILES,NAME")
    if input_ != "":
        i = 1
        SMILES, NAME = input_.split(",")[0], input_.split(",")[1]
        if ".[Cl-]" not in SMILES:
            SMILES += ".[Cl-]"
            pipeline(SMILES, NAME, cores=8, n_atoms=len(Chem.MolFromSmiles(SMILES).GetAtoms()), charge=0, multiplicity=1)
            with_ce, _, __ = make_best_ensemble(path_with=NAME)
            print(f"\tInitial Ensemble Size: {with_ce.n_conformers}", flush=True)
            # Prune ensemble
            if with_ce.n_conformers > 100:
                rmsd_thresh = 2.0
                energy_thresh = 1.0
            else:
                rmsd_thresh = 0.5
                energy_thresh = 2.0
            with_ce = trim_conformers(with_ce, rmsd_thresh=rmsd_thresh, energy_thresh=energy_thresh)
            print(f"\tPruned Ensemble Size: {with_ce.n_conformers}", flush=True)
            # Dump conformers
            dump_conformers(NAME, with_ce, i)

def gather_data_dicts(original_file, stoned_file):

    # Original Data
    df, cols = load_descriptors(original_file)
    original_data_dict = scale_data(df)
    # Stoned Data Scaled based on Original Data
    stoned_df, cols = load_descriptors(stoned_file)
    stoned_data_dict = scale_data(stoned_df, std_scaler=original_data_dict["scalers"]["std"], min_max_scaler=original_data_dict["scalers"]["mm"])
    ALL_SMILES = pd.concat([original_data_dict["original data"], stoned_data_dict["original data"]]).drop_duplicates(subset="SMILES")["SMILES"].tolist()
    ALL_df, _ = pd.concat([original_data_dict["original data"], stoned_data_dict["original data"]]).drop_duplicates(subset="SMILES"), cols
    all_data_dict = scale_data(ALL_df)
    return original_data_dict, stoned_data_dict, all_data_dict, ALL_SMILES, cols

def show_data(df):

    st.dataframe(df)
    fig, _ = histogram(df, bins=20, size=(10, 10))
    fig.tight_layout()
    st.pyplot(fig=fig)

def main(original_file, stoned_file):
    # Title
    st. set_page_config(layout="wide")
    st.title("PCA on MORFEUS Descriptors for Chemical Space of Amine Salts")
    # Buttons 
    raw_data = st.button("Raw Data")
    std_scale_data = st.button("Standardize Data")
    min_scale_data = st.button("MinMax Scale Data")
    view_pca_std = st.button("View PCA on Standard Scaled Data")
    view_pca_mm = st.button("View PCA on MinMax Scaled Data")
    view_pca_all = st.button("View PCA on All Data (Original + STONED)")
    # Calculate Descriptors on the fly....TODO
    input_smiles()
    # Get data dicts and SMILES
    og_data_dict, gen_data_dict, all_data_dict, all_smiles, cols = gather_data_dicts(original_file, stoned_file)
    if raw_data:
        tab1, tab2 = st.tabs(["Original Data", "Original + STONED"])
        with tab1:
            show_data(og_data_dict["original data"])
        with tab2:
            show_data(gen_data_dict["original data"])
    if std_scale_data:
        tab1, tab2 = st.tabs(["Original Data", "Original + STONED"])
        with tab1:
            show_data(og_data_dict["std scaled data"])
        with tab2:
            df = pd.concat([og_data_dict["std scaled data"], gen_data_dict["std scaled data"]])
            show_data(df)
    if min_scale_data:
        tab1, tab2 = st.tabs(["Original Data", "Original + STONED"])
        with tab1:
            show_data(og_data_dict["mm scaled data"])
        with tab2:
            df = pd.concat([og_data_dict["mm scaled data"], gen_data_dict["mm scaled data"]])
            show_data(df)
    if view_pca_std:
        tab1, tab2 = st.tabs(["Original Data", "Original + STONED"])
        with tab1:
            pca1, pca2, pca3 = st.tabs(["PC1", "PC2", "PC3"])
            pcas_df, exp_var, ifs, pca = run_PCA(og_data_dict["std scaled data"], cols, 3)
            with pca1:
                fig_1 = pca_3d(pcas_df, ifs, og_data_dict["std scaled data"], exp_var, label="PC1")
                st.plotly_chart(fig_1, theme=None, use_container_width=True)
            with pca2:
                fig_2 = pca_3d(pcas_df, ifs, og_data_dict["std scaled data"], exp_var, label="PC2")
                st.plotly_chart(fig_2, theme=None, use_container_width=True)
            with pca3:
                fig_3 = pca_3d(pcas_df, ifs, og_data_dict["std scaled data"], exp_var, label="PC3")
                st.plotly_chart(fig_3, theme=None, use_container_width=True)
        with tab2:
            print(gen_data_dict["std scaled data"])
            stoned_pcas_df = run_PCA(gen_data_dict["std scaled data"], cols, 3, pca)
            fig_1 = pca_3d(pcas_df, ifs, og_data_dict["std scaled data"], exp_var, label="PC1", color_by_dataset=stoned_pcas_df)
            st.plotly_chart(fig_1, theme=None, use_container_width=True)
    if view_pca_mm:
        tab1, tab2 = st.tabs(["Original Data", "Original + STONED"])
        with tab1:
            pca1, pca2, pca3 = st.tabs(["PC1", "PC2", "PC3"])
            pcas_df, exp_var, ifs, pca = run_PCA(og_data_dict["mm scaled data"], cols, 3)
            with pca1:
                fig_1 = pca_3d(pcas_df, ifs, og_data_dict["mm scaled data"], exp_var, label="PC1")
                st.plotly_chart(fig_1, theme=None, use_container_width=True)
            with pca2:
                fig_2 = pca_3d(pcas_df, ifs, og_data_dict["mm scaled data"], exp_var, label="PC2")
                st.plotly_chart(fig_2, theme=None, use_container_width=True)
            with pca3:
                fig_3 = pca_3d(pcas_df, ifs, og_data_dict["mm scaled data"], exp_var, label="PC3")
                st.plotly_chart(fig_3, theme=None, use_container_width=True)
        with tab2:
            stoned_pcas_df = run_PCA(gen_data_dict["mm scaled data"], cols, 3, pca)
            fig_1 = pca_3d(pcas_df, ifs, og_data_dict["mm scaled data"], exp_var, label="PC1", color_by_dataset=stoned_pcas_df)
            st.plotly_chart(fig_1, theme=None, use_container_width=True)
    if view_pca_all:
        tab1, tab2 = st.tabs(["Standard Scale", "MinMax Scale"])
        with tab1:
            pcas_df, exp_var, ifs, pca = run_PCA(all_data_dict["std scaled data"], cols, 3)
            pca1, pca2, pca3 = st.tabs(["PC1", "PC2", "PC3"])
            with pca1:
                fig_1 = pca_3d(pcas_df, ifs, all_data_dict["std scaled data"], exp_var, label="PC1")
                st.plotly_chart(fig_1, theme=None, use_container_width=True)
            with pca2:
                fig_2 = pca_3d(pcas_df, ifs, all_data_dict["std scaled data"], exp_var, label="PC2")
                st.plotly_chart(fig_2, theme=None, use_container_width=True)
            with pca3:
                fig_3 = pca_3d(pcas_df, ifs, all_data_dict["std scaled data"], exp_var, label="PC3")
                st.plotly_chart(fig_3, theme=None, use_container_width=True)
        with tab2:
            pcas_df, exp_var, ifs, pca = run_PCA(all_data_dict["mm scaled data"], cols, 3)
            pca1, pca2, pca3 = st.tabs(["PC1", "PC2", "PC3"])
            with pca1:
                fig_1 = pca_3d(pcas_df, ifs, all_data_dict["mm scaled data"], exp_var, label="PC1")
                st.plotly_chart(fig_1, theme=None, use_container_width=True)
            with pca2:
                fig_2 = pca_3d(pcas_df, ifs, all_data_dict["mm scaled data"], exp_var, label="PC2")
                st.plotly_chart(fig_2, theme=None, use_container_width=True)
            with pca3:
                fig_3 = pca_3d(pcas_df, ifs, all_data_dict["mm scaled data"], exp_var, label="PC3")
                st.plotly_chart(fig_3, theme=None, use_container_width=True)

    # TODO: visualize structures on hover
    #if st.button("experimental"):
    #    data = pd.concat([df, stoned_df.dropna()]).reset_index().drop("index", axis=1).drop_duplicates()
    #    print(data)
    #    data_std, std_scaler = std_scale(data, scaler=None)
    #    data_mm, mm_scaler = min_max_scale(data, scaler=None)
    #    pcas_df, exp_var, ifs, pca = run_PCA(data_mm, cols, 3)
    #    smiles = list(set(orig_smiles + stoned_smiles))
    #    p = plot_with_hover(pcas_df, ifs, data, exp_var, imgs=smiles)
    #    st.bokeh_chart(p)

main(sys.argv[1], sys.argv[2])
