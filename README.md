# Tachnical Test - Himydata

## Project Overview

This project analyzes products data to identify natural groupings of products based on sensor readings during production and physical/chemical properties.

The analysis pipeline consists of two major components:
1. **Exploratory Data Analysis**: Processing time series sensor data and product information
2. **Clustering Analysis**: Applying and comparing multiple clustering algorithms to identify product groupings

## Repository Structure

```
├── data/
│   ├── timeseries/      # Time series files with sensor readings
│   ├── combined_data.csv # Processed dataset for clustering
│   └── info_produits.txt # Product information with specifications
│
├── notebooks/
│   ├── 01_data_exploration.ipynb  # Initial data exploration notebook
│   └── 02_clustering_analysis.ipynb # Clustering implementation notebook
│
├── results/              # Results directory for outputs
│
├── scripts/
│   ├── run_exploration.py  # Script for data exploration
│   └── run_clustering.py   # Script for clustering analysis
│
├── src/
│   ├── clustering/        # Clustering algorithm implementations
│   │   ├── __init__.py
│   │   ├── compare_clustering.py  # Clustering comparison utilities
│   │   ├── dbscan.py      # DBSCAN implementation
│   │   ├── gmm.py         # Gaussian Mixture Model implementation
│   │   └── kmeans.py      # K-means implementation
│   │
│   ├── data_processing/   # Data handling modules
│   │   ├── __init__.py
│   │   ├── loading.py     # Data loading functions
│   │   └── preprocessing.py # Data preprocessing functions
│   │
│   ├── feature_engineering/ # Feature extraction modules
│   │   ├── __init__.py
│   │   └── feature_extraction.py # Feature extraction from time series
│   │
│   └── visualization/     # Visualization modules
│       ├── __init__.py
│       └── exploratory_visualization.py # Visualization functions when ecxploring the data
│       └── cluster_visualization.py # Visualization functions after clustering
│
├── venv/                 # Virtual environment (not tracked by git)
│
├── main.py               # Main entry point for the complete pipeline
├── setup.py              # Package installation and entry points configuration 
├── README.md             # Project documentation
└── requirements.txt      # Project dependencies
```

## Dataset Description

The analysis works with two primary data sources:

### 1. Product Information (`info_produits.txt`)
Contains physical and chemical properties of products:
- Product identifiers
- Physical dimensions (thickness, width)
- Chemical composition (Carbon, Chrome, Manganese, Silicon, etc.)

### 2. Time Series Sensor Data (`timeseries/`)
Contains sensor readings during manufacturing:
- One file per product (named by product ID)
- 50 readings per second from various sensors
- Target column: "Pinceur Sup Mesure de courant"

## Features and Analysis Approach

### Time Series Feature Extraction
From the sensor data, we extract multiple features capturing different aspects of the manufacturing process:

#### 1. Statistical Features
- **Mean**: Captures the central tendency of sensor readings
- **Standard Deviation**: Measures process variability and stability
- **Minimum & Maximum Values**: Identifies operational boundaries
- **Range**: Quantifies the overall amplitude of sensor fluctuations during manufacturing

#### 2. State-Based Features
Our analysis revealed that the manufacturing process exhibits distinct operational states. We characterized these states with specialized features:

- **High-State Mean**: Average reading during the high-amplitude phase, indicating maximum operational intensity
- **Low-State Mean**: Average reading during the low-amplitude phase, representing baseline operational levels
- **High-State Ratio**: Proportion of time spent in high state, reflecting process duration allocation
- **High-State Standard Deviation**: Stability measure during high-intensity operations
- **Low-State Standard Deviation**: Stability measure during baseline operations

This bimodal characterization allows us to quantify differences in how each product behaves during different manufacturing phases.

#### 3. Transition Features
Manufacturing transitions between states contain critical information about process dynamics:

- **Transition Point**: Normalized timing of the major state change (0-1 scale), revealing process phase distribution
- **Transition Magnitude**: Size of the most significant drop in readings, indicating transition abruptness

These features capture how unexpectedly the process changes and at what point in the manufacturing cycle.

#### 4. Trend Analysis
- **Trend Slope**: Linear regression coefficient measuring the overall directional tendency of the process.

### Exploratory Findings

Our exploratory analysis revealed several key insights:

1. **Product Dimensions**: 
   - Thickness distribution shows distinct peaks at 2.0mm and 3.5mm
   - Width clustering around standardized values (1000mm, 1200mm, 1500mm)
   - No strong correlation between thickness and width

2. **Chemical Composition**:
   - Manganese has highest concentration and variability
   - Carbon, Chrome, and Silicon show more consistent levels

3. **Manufacturing Patterns**:
   - All products follow a similar cycle: rise → high plateau → decline → low plateau
   - Significant differences in high-state and low-state characteristics
   - Transition points vary in timing and magnitude

## Clustering Methodology

We implemented and compared three advanced clustering algorithms to identify natural groupings in the data:

### 1. K-means Clustering
- **Approach**: Partition-based algorithm using Euclidean distance
- **Implementation**: Optimized for finding compact, spherical clusters
- **Parameter Selection**: Optimal cluster number determined through silhouette score, inertia, Davies-Bouldin index, and Calinski-Harabasz index

### 2. Gaussian Mixture Models (GMM)
- **Approach**: Probabilistic model using multivariate Gaussian distributions
- **Implementation**: Tested four covariance structures (full, tied, diagonal, spherical)
- **Key Feature**: Provides membership probabilities for each data point
- **Results**: "Tied" covariance performed best, indicating clusters share similar internal variability patterns

### 3. DBSCAN (Density-Based Spatial Clustering)
- **Approach**: Density-based algorithm that identifies non-spherical clusters
- **Implementation**: Parameters estimated through k-distance analysis
- **Key Feature**: Automatically identifies outliers as noise points
- **Parameter Selection**: Epsilon=5.0381 determined at the optimal density boundary

## Key Results

Our analysis identified two primary clusters of products with distinct characteristics:

### Cluster 1 (225 products):
- Higher sensor readings (mean, standard deviation, max, range)
- Higher high-state metrics and lower minimum values
- Higher levels of most chemical elements except Aluminum, Copper, and Phosphore

### Cluster 2 (219 products):
- Lower sensor readings overall but higher minimum values
- More negative trend slope
- Higher Aluminum, Copper, Phosphore, and Soufre content

K-means and GMM algorithms produced nearly identical clusters (silhouette score: 0.370), validating the natural grouping in the data. DBSCAN offered a different perspective, identifying one dominant manufacturing group (428 products) with a small number of outliers (11 noise points).

The clustering validation indicated K-means as technically the best method based on silhouette score (0.3702), with GMM close behind (0.3696).

## Installation and Requirements
1. Clone the repository
2. Create and activate a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
3. Install required dependencies
```bash
pip install -r requirements.txt
```
Required packages:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- scipy
- kneed (for DBSCAN parameter estimation)

4. Install in development mode
```bash
pip install -e .
```
This development mode installation automatically adds the project to your Python path and makes the command-line tools available.

If you're not using the development installation, you'll need to set PYTHONPATH:

```bash
# Windows (PowerShell)
$env:PYTHONPATH = "Path\to\himydata"

# Windows (Command Prompt)
set PYTHONPATH=Path\to\himydata

# Linux/macOS
export PYTHONPATH=/path/to/himydata
```

## Usage

### Running the Complete Pipeline

The main entry point provides access to the full analysis pipeline:

```bash
python main.py
```

### Interactive Mode

For step-by-step analysis with user interaction:

```bash
python main.py --interactive
```

This will display a menu allowing you to:
- Explore product information
- Analyze individual time series
- Compare multiple products
- Run different clustering algorithms
- Compare clustering results

### Running Specific Components

#### Data Exploration Only

```bash
python main.py --mode exploration
```

For interactive exploration with menu options:

```bash
python scripts/run_exploration.py --interactive
```

#### Clustering Analysis Only

```bash
python main.py --mode clustering
```

With custom data file:

```bash
python main.py --mode clustering --data path/to/combined_data.csv
```

For interactive clustering with parameter selection:

```bash
python scripts/run_clustering.py --interactive
```
## Usage

### Running the Complete Pipeline

The main entry point provides access to the full analysis pipeline:

```bash
python main.py
```

### Interactive Mode

For step-by-step analysis with user interaction:

```bash
python main.py --interactive
```

This will display a menu allowing you to:
- Explore product information
- Analyze individual time series
- Compare multiple products
- Run different clustering algorithms
- Compare clustering results

### Running Specific Components

#### Data Exploration Only

```bash
python main.py --mode exploration
```

For interactive exploration with menu options:

```bash
python scripts/run_exploration.py --interactive
```

#### Clustering Analysis Only

```bash
python main.py --mode clustering
```

With custom data file:

```bash
python main.py --mode clustering --data path/to/combined_data.csv
```

For interactive clustering with parameter selection:

```bash
python scripts/run_clustering.py --interactive
```

**NB:** For better comprehension of the data and analytical process, we recommend running **the interactive mode**. It provides guided workflows with detailed explanations of each step and allows you to explore different aspects of the data and clustering methods at your own pace. 
This approach is particularly valuable for first-time users to understand the dataset characteristics and the reasoning behind parameter selections.

### Jupyter Notebooks

The project includes two Jupyter notebooks that document the exploration and clustering process:

#### 1. Data Exploration (`01_data_exploration.ipynb`)
This notebook contains the initial data exploration process, including:
- Loading and visualization of product information data
- Analysis of time series sensor readings
- Exploration of manufacturing patterns across different products
- Feature extraction from time series data
- Creation of the combined dataset used for clustering

#### 2. Clustering Analysis (`02_clustering_analysis.ipynb`)
This notebook implements and evaluates different clustering approaches:
- Data preprocessing and scaling
- PCA for dimensionality reduction
- K-means clustering implementation and visualization
- Gaussian Mixture Model (GMM) analysis with covariance type selection
- DBSCAN implementation with parameter estimation
- Comparative analysis of clustering results

These notebooks provide a detailed, step-by-step walkthrough of the analysis process with explanatory text, visualizations, and code. They can be used to understand the methodology and reasoning behind the modular implementation in the `src` directory.

## Technical Implementation Details

### Data Processing Architecture
- **Modular Design**: Clear separation of concerns between data loading, preprocessing, feature extraction, and visualization
- **Efficient Pipeline**: Optimized processing flow minimizes redundant operations
- **Error Handling**: Robust error management for missing data and outliers

### Clustering Algorithm Implementation
- **K-means**: Implemented with multiple initialization strategies and silhouette optimization
- **GMM**: Comprehensive covariance structure testing (full, tied, diagonal, spherical)
- **DBSCAN**: Automated parameter estimation using k-distance analysis

### Interactive Analysis Capabilities
- **Menu-Driven Interface**: Intuitive navigation through analysis options
- **Parameter Customization**: User control over key algorithm parameters
- **Step-by-Step Workflow**: Guided process through the complete analysis pipeline


## Future Work

This analysis provides a foundation for several valuable extensions:

1. **Supervised Learning**: Predictive models based on the identified clusters
2. **Time Series Forecasting**: Early prediction of manufacturing outcomes 
3. **Anomaly Detection**: Real-time quality control monitoring
4. **Feature Importance Analysis**: Identifying key drivers of cluster separation
5. **Hierarchical Clustering**: Exploring sub-clusters for more granular insights