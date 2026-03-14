import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Professional Data Face (HTML/CSS)
header_html = """
<div style="background-color: #f0f7da; border-radius: 15px; padding: 30px; border: 2px solid #5d8233; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
    <h1 style="color: #2e4c19; text-align: center; font-size: 3em; margin-bottom: 5px;">🌾 Rice Production Plan 2025</h1>
    <h3 style="color: #5d8233; text-align: center; font-size: 1.5em; font-style: italic;">Strategic Analysis of Punjab's Agricultural Targets</h3>
    <hr style="border: 1px solid #5d8233;">
    <div style="display: flex; justify-content: space-around; margin-top: 20px;">
        <div style="text-align: center;">
            <p style="font-weight: bold; color: #2e4c19; margin-bottom: 0;">Author</p>
            <p style="color: #5d8233; margin-top: 5px;">Muhammad Usman</p>
            <p style="font-size: 0.8em; color: #777;">Scientific Officer</p>
        </div>
        <div style="text-align: center;">
            <p style="font-weight: bold; color: #2e4c19; margin-bottom: 0;">Dataset</p>
            <p style="color: #5d8233; margin-top: 5px;">Rice Punjab 2025</p>
            <p style="font-size: 0.8em; color: #777;">Official Govt Targets</p>
        </div>
        <div style="text-align: center;">
            <p style="font-weight: bold; color: #2e4c19; margin-bottom: 0;">Last Updated</p>
            <p style="color: #5d8233; margin-top: 5px;">February 2025</p>
        </div>
    </div>
</div>
"""

# 1. Introduction Cell
intro_md = """
# 1. Introduction
Welcome to the Exploratory Data Analysis (EDA) of the **Punjab Rice Production Plan 2025**. This notebook analyzes the strategic targets set by the Department of Agriculture for the upcoming Kharif season.

### Key Objectives:
*   Identify high-priority districts for **Basmati** vs **Coarse** rice.
*   Analyze the relationship between **Area Targets** and **Production Goals**.
*   Evaluate the adoption of **Direct Seeding of Rice (DSR)** across different divisions.
*   Visualize the diversity of **Recommended Varieties**.
"""

# 2. Setup Cell
setup_code = """
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Setting default theme
pio.templates.default = "plotly_white"

# Load the dataset
df = pd.read_csv('Rice_Production_Plan_2025.csv')
df.head()
"""

# 3. Data Overview
data_overview_md = """
# 2. Data Overview & Structure
Before diving into visualizations, let's understand the data types and basic statistics.
"""

data_overview_code = """
# Displaying basic info
print(f"Dataset Shape: {df.shape}")
df.info()

# Basic descriptive statistics for numerical columns
df.describe().T.style.background_gradient(cmap='Greens')
"""

# 4. Regional Analysis (Plotly)
regional_analysis_md = """
# 3. Regional Distribution of Rice Cultivation
In this section, we analyze how rice cultivation targets are distributed across different divisions and districts.
"""

regional_analysis_code = """
# Treemap of Area Targets by Division and District
fig_treemap = px.treemap(df, 
                         path=['Division', 'District'], 
                         values='Area_Target_Acres',
                         color='Rice_Type',
                         title='Hierarchical Distribution of Rice Cultivation Area (Acres)',
                         color_discrete_map={'Basmati': '#5d8233', 'Coarse': '#a4c639'})

fig_treemap.update_layout(margin=dict(t=50, l=25, r=25, b=25))
fig_treemap.show()
"""

# 5. Production Targets Analysis
prod_analysis_md = """
# 4. Production Targets: Basmati vs Coarse
Basmati is the backbone of Pakistan's agricultural exports. Let's see how production targets differ.
"""

prod_analysis_code = """
# Bar chart for total production by rice type
fig_bar = px.bar(df.groupby('Rice_Type')[['Production_Target_Tons']].sum().reset_index(),
                 x='Rice_Type', y='Production_Target_Tons',
                 color='Rice_Type',
                 text_auto='.2s',
                 title='Total Production Target (Tons) by Rice Category',
                 color_discrete_map={'Basmati': '#2e4c19', 'Coarse': '#8db600'})

fig_bar.show()

# Scatter plot: Area vs Production
fig_scatter = px.scatter(df, x='Area_Target_Acres', y='Production_Target_Tons',
                         color='Rice_Type', size='Yield_Target_Maunds_Per_Acre',
                         hover_name='District',
                         title='Relationship between Area (Acres) and Production (Tons)',
                         labels={'Area_Target_Acres': 'Cultivation Area', 'Production_Target_Tons': 'Production Target'},
                         trendline="ols")

fig_scatter.show()
"""

# 6. Sowing Methods & Varieties
method_analysis_md = """
# 5. Modern Implementation: Sowing Methods & Varieties
The 2025 plan emphasizes the transition to **Direct Seeding of Rice (DSR)** to conserve water.
"""

method_analysis_code = """
# Sunburst chart for Sowing Methods and Varieties
fig_sunburst = px.sunburst(df, path=['Sowing_Method', 'Rice_Type', 'Recommended_Variety'], 
                           values='Area_Target_Acres',
                           title='Cultivation Method & Variety Strategy',
                           color_discrete_sequence=px.colors.sequential.Greens_r)

fig_sunburst.show()
"""

# 7. Summary & Interpretation
summary_md = """
# 6. Final Interpretation & Insights

Based on our analysis of the **Production Plan Rice 2025**, here are the key takeaways:

1.  **Core Basmati Regions**: Gujranwala and Lahore divisions remain the primary hubs for Basmati rice, with the highest area targets assigned to districts like **Gujranwala** and **Sheikhupura**.
2.  **Yield Efficiency**: Coarse rice varieties (IRRI series) show significantly higher yield targets (up to 60 maunds/acre) compared to Basmati, making them crucial for domestic food security.
3.  **Modernization**: There is a clear strategic push toward **Direct Seeding of Rice (DSR)** in specific districts (e.g., Okara, Multan) to mitigate water scarcity issues.
4.  **Variety Diversity**: The plan promotes a diverse range of varieties (Super Basmati, Kainat 1121, KSK-133) to cater to both high-end export markets and localized high-yield requirements.

---
### **Connect with the Author**
This notebook was developed by **Muhammad Usman, Scientific Officer**. 
For any queries related to agricultural data analysis or the Punjab Rice Sector, feel free to reach out.

*Note: This analysis is based on the 2025 targets and should be used as a reference for agricultural planning and research.*
"""

# Adding all cells to notebook
nb.cells = [
    nbf.v4.new_markdown_cell(header_html),
    nbf.v4.new_markdown_cell(intro_md),
    nbf.v4.new_code_cell(setup_code),
    nbf.v4.new_markdown_cell(data_overview_md),
    nbf.v4.new_code_cell(data_overview_code),
    nbf.v4.new_markdown_cell(regional_analysis_md),
    nbf.v4.new_code_cell(regional_analysis_code),
    nbf.v4.new_markdown_cell(prod_analysis_md),
    nbf.v4.new_code_cell(prod_analysis_code),
    nbf.v4.new_markdown_cell(method_analysis_md),
    nbf.v4.new_code_cell(method_analysis_code),
    nbf.v4.new_markdown_cell(summary_md)
]

# Write to file
nb_path = r'e:\AI_Chilla\AI_CHILLA_2026\Kaggle_datasets\Rice\Rice_Analysis_2025.ipynb'
with open(nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook successfully created at {nb_path}")
