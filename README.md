
# Supply Chain Optimization App

A Streamlit web application for optimizing supply chain logistics and visualizing facility locations using interactive maps.

## Features

- Interactive map visualization of plants and warehouses across the US
- Supply chain network optimization using CVXPY
- Geolocation services integration with Nominatim
- Route visualization using Folium and AntPath
- Product inventory management across multiple facilities

## Dependencies

- streamlit
- folium
- cvxpy
- geopy
- matplotlib
- numpy
- pandas

## Installation

```sh
pip install -r requirements.txt
```

Run the Streamlit app locally:

```sh
streamlit run app.py
```

## Usage

### Project Structure

- `app.py` - Main application file containing the Streamlit interface and optimization logic
- `mappings` - Data dictionaries containing:
  - Products (12 oz cans, lids, etc.)
  - Plants (manufacturing facilities)
  - Warehouses (distribution centers)

### Optimization Workflow

The optimization workflow in the Streamlit app is organized into tabs for a step-by-step process:

#### 1. **Load and Clean Data**
   - **Tab**: "Data Loader"
   - **Description**: Upload your Excel file containing optimization data.
   - **Actions**:
     - Choose the appropriate sheet for the month you want to analyze (e.g., `Solver-Jan`).
     - The app automatically validates the file and cleans the data by filtering non-zero demand and filling missing values.

#### 2. **Check Feasibility**
   - **Tab**: "Feasibility Check"
   - **Description**: Evaluate whether the production and shipping capacities are sufficient to meet demand.
   - **Actions**:
     - View tables that summarize demand, capacity, and feasibility for each product.
     - Color-coded status shows whether each product is feasible (`green`) or infeasible (`red`).

#### 3. **Generate Visualizations**
   - **Tab**: "Visualization"
   - **Description**: Visualize production and shipping patterns using maps and tables.
   - **Actions**:
     - View a production capability map (`■ = can produce, □ = cannot produce`).
     - View a shipping demand map (`■ = has demand, □ = no demand`).

#### 4. **Adjust Capacities**
   - **Tab**: "Capacity Adjustment"
   - **Description**: Automatically adjust production capacities to ensure feasibility.
   - **Actions**:
     - The app identifies over-allocated capacities and proportionally adjusts them.
     - View adjustments made to capacities and their scaling factors.

#### 5. **Apply Constraints**
   - **Tab**: "Constraints"
   - **Description**: Define and apply sourcing constraints based on business rules or seasonal requirements.
   - **Actions**:
     - Constraints include minimum sourcing from specific plants and warehouses.
     - Seasonal constraints are applied automatically for months like September to December.

#### 6. **Optimize Costs**
   - **Tab**: "Optimization"
   - **Description**: Minimize total costs using CVXPY while satisfying all demand and capacity constraints.
   - **Actions**:
     - The app calculates optimal production and shipping allocations.
     - View cost breakdowns for production and shipping.
     - Compare total costs for the current setup versus optimized setups.

### Additional Information
- **Error Handling**: The app provides detailed error messages for invalid inputs or constraints.
- **Performance Tracking**: The app displays key metrics such as total savings, capacity utilization, and optimization statuses.

For further technical details, refer to the `app.py` file.
