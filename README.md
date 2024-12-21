# Tellco Data Analysis

This project analyzes telecommunications data to derive insights on customer engagement metrics. It includes exploratory data analysis (EDA), correlation analysis, dimensionality reduction, and customer clustering using K-Means.

## Project Structure

tellco-data-analysis/
├── .github/
  ├── workflows/
    ├── unittests.yml
├── .vscode/
  ├── settings.json
├── data/
  ├── telecom.sql
├── scripts/
│ ├── __init__.py
│ ├── dimensionality_reduction.py
│ ├── aggr_metrics.py
| ├── bivariate_analysis.py
| ├── data_clean.py
| ├── data_loader.py
| ├── engagement_analysis.py
| ├── graphical_univariate_analysis.py
| ├── non_graphical_univariate_analysis.py
| ├── README.md
| ├── task1_2.py
| ├── variable_transformations.py
├── notebooks/
  ├── __init__.py
  ├── analysis.ipynb
  ├── engagement_anal.ipynb
  ├── README.md
├── requirements.txt
└── .env
├── .gitignore
└── src
    ├── __init__.py
└── tests
    ├── __init__.py
└── telvenv


- **scripts/**: Contains Python scripts for different analyses.
- **notebooks/**: Jupyter notebooks demonstrating the analysis steps.
- **requirements.txt**: Lists the Python packages required for the project.

## Script Functionalities

- **aggr_metrics.py**: Aggregates various metrics for comprehensive analysis.
- **bivariate_analysis.py**: Conducts bivariate analysis to explore relationships between pairs of variables.
- **engagement_analysis.py**: Normalizes customer metrics, applies K-Means clustering, and generates engagement insights.
- **data_clean.py**: Cleans and preprocesses data extracted from the `xdr_data`.
- **data_loader.py**: Loads data from the TelecomDB database for analysis.
- **graphical_univariate_analysis.py**: Performs graphical analysis of univariate distributions.
- **non_graphical_univariate.py**: Conducts non-graphical analysis of univariate data.
- **task1_2.py**: Executes all tasks outlined in Task 1.2.
- **variable_transformation.py**: Transforms specified variables for further analysis.

## Commit Message Guidelines

When making changes to the repository, please follow these guidelines for commit messages:
- Use the imperative mood ("Add feature" instead of "Added feature").
- Be concise but descriptive.
- Include a reference to any relevant issue or task.

## Usage

### Run the Analysis

1. **Load and Preprocess Data**
   - Use `data_loader.py` to load data from the TelecomDB database.
   - Clean the data using `data_clean.py` to ensure it's ready for analysis.

2. **Exploratory Data Analysis (EDA)**
   - Perform univariate analysis with `graphical_univariate_analysis.py` for visual insights and `non_graphical_univariate.py` for statistical summaries.
   - Conduct bivariate analysis using `bivariate_analysis.py` to explore relationships between variables.

3. **Dimensionality Reduction**
   - Implement dimensionality reduction techniques in the appropriate script (e.g., PCA) to simplify the dataset while preserving important information.

4. **Engagement Metrics and Clustering**
   - Aggregate customer metrics with `aggr_metrics.py`.
   - Normalize metrics and apply K-Means clustering using `engagement_analysis.py`.
   - Execute `task1_2.py` to run all tasks related to this analysis step and derive actionable insights.

## Summary of Findings

- **Customer Segmentation**: The analysis identified three distinct customer segments based on engagement metrics, optimized through K-Means clustering.
- **Engagement Insights**: Key metrics such as session frequency, total duration, and download/upload volumes were critical in understanding customer behavior.
- **Correlation Analysis**: Significant correlations were found between various engagement metrics, providing insights into customer interactions with services.
- **Dimensionality Reduction**: Techniques like PCA were effective in reducing complexity while retaining essential information, aiding in better visualization and analysis.
- **Top Applications**: The analysis highlighted the top three most used applications, guiding potential areas for improvement and marketing focus.
- **Data Quality Improvement**: Data cleaning processes enhanced the overall quality of the dataset, resulting in more reliable analysis outcomes.