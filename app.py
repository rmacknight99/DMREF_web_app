import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
import plotly.graph_objects as go

def load_descriptors(in_file):

    descriptors = pd.read_csv(in_file)
    return descriptors, descriptors.columns

# Title
st.title("PCA applied to Chemical Space of Amine Salts")
df, cols = load_descriptors("descriptors_ORCA_13.csv")