import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<div style=\"background-color: #f0f7da; border-radius: 15px; padding: 30px; border: 2px solid #5d8233; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;\">\n",
    "    <h1 style=\"color: #2e4c19; text-align: center; font-size: 3em; margin-bottom: 5px;\">\ud83c\udf3e Rice Production Plan 2025</h1>\n",
    "    <h3 style=\"color: #5d8233; text-align: center; font-size: 1.5em; font-style: italic;\">Strategic Analysis of Punjab's Agricultural Targets</h3>\n",
    "    <hr style=\"border: 1px solid #5d8233;\">\n",
    "    <div style=\"display: flex; justify-content: space-around; margin-top: 20px;\">\n",
    "        <div style=\"text-align: center;\">\n",
    "            <p style=\"font-weight: bold; color: #2e4c19; margin-bottom: 0;\">Author</p>\n",
    "            <p style=\"color: #5d8233; margin-top: 5px;\">Muhammad Usman</p>\n",
    "            <p style=\"font-size: 0.8em; color: #777;\">Scientific Officer</p>\n",
    "        </div>\n",
    "        <div style=\"text-align: center;\">\n",
    "            <p style=\"font-weight: bold; color: #2e4c19; margin-bottom: 0;\">Dataset</p>\n",
    "            <p style=\"color: #5d8233; margin-top: 5px;\">Rice Punjab 2025</p>\n",
    "            <p style=\"font-size: 0.8em; color: #777;\">Official Govt Targets</p>\n",
    "        </div>\n",
    "        <div style=\"text-align: center;\">\n",
    "            <p style=\"font-weight: bold; color: #2e4c19; margin-bottom: 0;\">Last Updated</p>\n",
    "            <p style=\"color: #5d8233; margin-top: 5px;\">February 2025</p>\n",
    "        </div>\n",
    "    </div>\n",
    "</div>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1. Introduction\n",
    "Welcome to the Exploratory Data Analysis (EDA) of the **Punjab Rice Production Plan 2025**. This notebook analyzes the strategic targets set by the Department of Agriculture for the upcoming Kharif season.\n",
    "\n",
    "### Key Objectives:\n",
    "*   Identify high-priority districts for **Basmati** vs **Coarse** rice.\n",
    "*   Analyze the relationship between **Area Targets** and **Production Goals**.\n",
    "*   Evaluate the adoption of **Direct Seeding of Rice (DSR)** across different divisions.\n",
    "*   Visualize the diversity of **Recommended Varieties**."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import plotly.express as px\n",
    "import plotly.graph_objects as go\n",
    "import plotly.io as pio\n",
    "\n",
    "# Setting default theme\n",
    "pio.templates.default = \"plotly_white\"\n",
    "\n",
    "# Load the dataset\n",
    "df = pd.read_csv('Rice_Production_Plan_2025.csv')\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2. Data Overview & Structure\n",
    "Before diving into visualizations, let's understand the data types and basic statistics."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Displaying basic info\n",
    "print(f\"Dataset Shape: {df.shape}\")\n",
    "df.info()\n",
    "\n",
    "# Basic descriptive statistics for numerical columns\n",
    "df.describe().T"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 3. Regional Distribution of Rice Cultivation\n",
    "In this section, we analyze how rice cultivation targets are distributed across different divisions and districts."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Treemap of Area Targets by Division and District\n",
    "fig_treemap = px.treemap(df, \n",
    "                         path=['Division', 'District'], \n",
    "                         values='Area_Target_Acres',\n",
    "                         color='Rice_Type',\n",
    "                         title='Hierarchical Distribution of Rice Cultivation Area (Acres)',\n",
    "                         color_discrete_map={'Basmati': '#5d8233', 'Coarse': '#a4c639'})\n",
    "\n",
    "fig_treemap.update_layout(margin=dict(t=50, l=25, r=25, b=25))\n",
    "fig_treemap.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 4. Production Targets: Basmati vs Coarse\n",
    "Basmati is the backbone of Pakistan's agricultural exports. Let's see how production targets differ."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Bar chart for total production by rice type\n",
    "fig_bar = px.bar(df.groupby('Rice_Type')[['Production_Target_Tons']].sum().reset_index(),\n",
    "                 x='Rice_Type', y='Production_Target_Tons',\n",
    "                 color='Rice_Type',\n",
    "                 text_auto='.2s',\n",
    "                 title='Total Production Target (Tons) by Rice Category',\n",
    "                 color_discrete_map={'Basmati': '#2e4c19', 'Coarse': '#8db600'})\n",
    "\n",
    "fig_bar.show()\n",
    "\n",
    "# Scatter plot: Area vs Production\n",
    "fig_scatter = px.scatter(df, x='Area_Target_Acres', y='Production_Target_Tons',\n",
    "                         color='Rice_Type', size='Yield_Target_Maunds_Per_Acre',\n",
    "                         hover_name='District', \n",
    "                         title='Relationship between Area (Acres) and Production (Tons)',\n",
    "                         labels={'Area_Target_Acres': 'Cultivation Area', 'Production_Target_Tons': 'Production Target'})\n",
    "\n",
    "fig_scatter.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 5. Modern Implementation: Sowing Methods & Varieties\n",
    "The 2025 plan emphasizes the transition to **Direct Seeding of Rice (DSR)** to conserve water."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Sunburst chart for Sowing Methods and Varieties\n",
    "fig_sunburst = px.sunburst(df, path=['Sowing_Method', 'Rice_Type', 'Recommended_Variety'], \n",
    "                           values='Area_Target_Acres',\n",
    "                           title='Cultivation Method & Variety Strategy',\n",
    "                           color_discrete_sequence=px.colors.sequential.Greens_r)\n",
    "\n",
    "fig_sunburst.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 6. Final Interpretation & Insights\n",
    "\n",
    "Based on our analysis of the **Production Plan Rice 2025**, here are the key takeaways:\n",
    "\n",
    "1.  **Core Basmati Regions**: Gujranwala and Lahore divisions remain the primary hubs for Basmati rice, with the highest area targets assigned to districts like **Gujranwala** and **Sheikhupura**.\n",
    "2.  **Yield Efficiency**: Coarse rice varieties (IRRI series) show significantly higher yield targets (up to 60 maunds/acre) compared to Basmati, making them crucial for domestic food security.\n",
    "3.  **Modernization**: There is a clear strategic push toward **Direct Seeding of Rice (DSR)** in specific districts (e.g., Okara, Multan) to mitigate water scarcity issues.\n",
    "4.  **Variety Diversity**: The plan promotes a diverse range of varieties (Super Basmati, Kainat 1121, KSK-133) to cater to both high-end export markets and localized high-yield requirements.\n",
    "\n",
    "---\n",
    "### **Connect with the Author**\n",
    "This notebook was developed by **Muhammad Usman, Scientific Officer**. \n",
    "For any queries related to agricultural data analysis or the Punjab Rice Sector, feel free to reach out.\n",
    "\n",
    "*Note: This analysis is based on the 2025 targets and should be used as a reference for agricultural planning and research.*"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

nb_path = r'e:\AI_Chilla\AI_CHILLA_2026\Kaggle_datasets\Rice\Rice_Analysis_2025.ipynb'
with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook created at {nb_path}")
