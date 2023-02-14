import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
import plotly.graph_objects as go


# Title
st.title("PCA applied to Chemical Space of Amine Salts")
