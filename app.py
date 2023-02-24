from app_utils import *
#from pca_utils import *
from desc_utils import *

import sys

def main(original_file, stoned_file):
    # Title
    st. set_page_config(layout="wide")
    st.title("PCA on MORFEUS Descriptors for Chemical Space of Amine Salts")
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
    df, cols = load_descriptors(original_file)
    df_std_scaled, std_scaler = std_scale(df)
    df_mm_scaled, mm_scaler = min_max_scale(df)
    raw_data = st.button("Raw Data")
    std_scale_data = st.button("Standardize Data")
    min_scale_data = st.button("MinMax Scale Data")
    stoned_df, cols = load_descriptors(stoned_file)
    stoned_std_df_scaled, std_scaler = std_scale(stoned_df, std_scaler)
    stoned_mm_df_scaled, mm_scaler = std_scale(stoned_df, mm_scaler)
    if raw_data:
        tab1, tab2 = st.tabs(["Original Data", "Original + STONED"])
        with tab1:
            st.dataframe(df)
            fig, _ = histogram(df, bins=20, size=(10, 10))
            fig.tight_layout()
            st.pyplot(fig=fig)
        with tab2:
            st.dataframe(stoned_df)
            fig, _ = histogram(stoned_df, bins=20, size=(10, 10))
            fig.tight_layout()
            st.pyplot(fig=fig)
    if std_scale_data:
        st.dataframe(df_std_scaled)
        tab1, tab2 = st.tabs(["Original Data", "Original + STONED"])
        with tab1:
            fig, _ = histogram(df_std_scaled, bins=20, size=(10, 10))
            fig.tight_layout()
            st.pyplot(fig=fig)
        with tab2:
            fig, _ = histogram(pd.concat([df_std_scaled, stoned_std_df_scaled]))
            fig.tight_layout()
            st.pyplot(fig=fig)
    if min_scale_data:
        st.dataframe(df_mm_scaled)
        tab1, tab2 = st.tabs(["Original Data", "Original + STONED"])
        with tab1:
            fig, _ = histogram(df_mm_scaled, bins=20, size=(10, 10))
            fig.tight_layout()
            st.pyplot(fig=fig)
        with tab2:
            fig, _ = histogram(pd.concat([df_mm_scaled, stoned_mm_df_scaled]), bins=20, size=(10, 10))
            fig.tight_layout()
            st.pyplot(fig=fig)
    if st.button("View PCA on Standard Scaled Data"):
        tab1, tab2 = st.tabs(["Original Data", "Original + STONED"])
        with tab1:
            pca1, pca2, pca3 = st.tabs(["PC1", "PC2", "PC3"])
            pcas_df, exp_var, ifs, pca = run_PCA(df_std_scaled, cols, 3)
            with pca1:
                fig_1 = pca_3d(pcas_df, ifs, df, exp_var, label="PC1")
                st.plotly_chart(fig_1, theme=None, use_container_width=True)
            with pca2:
                fig_2 = pca_3d(pcas_df, ifs, df, exp_var, label="PC2")
                st.plotly_chart(fig_2, theme=None, use_container_width=True)
            with pca3:
                fig_3 = pca_3d(pcas_df, ifs, df, exp_var, label="PC3")
                st.plotly_chart(fig_3, theme=None, use_container_width=True)
        with tab2:
            stoned_df, cols = load_descriptors(stoned_file)
            stoned_df_scaled, std_scaler = std_scale(stoned_df, std_scaler)
            stoned_pcas_df = run_PCA(stoned_df_scaled, cols, 3, pca)
            fig_1 = pca_3d(pcas_df, ifs, df, exp_var, label="PC1", color_by_dataset=stoned_pcas_df)
            st.plotly_chart(fig_1, theme=None, use_container_width=True)
    if st.button("View PCA on MinMax Scaled Data"):
        tab1, tab2 = st.tabs(["Original Data", "Original + STONED"])
        with tab1:
            pca1, pca2, pca3 = st.tabs(["PC1", "PC2", "PC3"])
            pcas_df, exp_var, ifs, pca = run_PCA(df_mm_scaled, cols, 3)
            with pca1:
                fig_1 = pca_3d(pcas_df, ifs, df, exp_var, label="PC1")
                st.plotly_chart(fig_1, theme=None, use_container_width=True)
            with pca2:
                fig_2 = pca_3d(pcas_df, ifs, df, exp_var, label="PC2")
                st.plotly_chart(fig_2, theme=None, use_container_width=True)
            with pca3:
                fig_3 = pca_3d(pcas_df, ifs, df, exp_var, label="PC3")
                st.plotly_chart(fig_3, theme=None, use_container_width=True)
        with tab2:
            stoned_df, cols = load_descriptors(stoned_file)
            dataset = ["blue" for _ in range(df.shape[0])] + ["red" for _ in range(stoned_df.shape[0])]
            stoned_df_scaled, mm_scaler = std_scale(stoned_df, mm_scaler)
            stoned_pcas_df = run_PCA(stoned_df_scaled, cols, 3, pca)
            fig_1 = pca_3d(pcas_df, ifs, df, exp_var, label="PC1", color_by_dataset=stoned_pcas_df)
            st.plotly_chart(fig_1, theme=None, use_container_width=True)

main(sys.argv[1], sys.argv[2])
