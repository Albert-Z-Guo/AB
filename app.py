import io
from collections import defaultdict
from typing import Tuple

import cvxpy as cp
import folium
from folium import plugins
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# Define mappings
mappings = {
    'products': {1: "12 oz", 2: "12 oz Sleek", 3: "16 oz", 4: "Lids", 5: "25 oz", 6: "16 BT"},
    'plants': {1: "MIR", 2: "JAX", 3: "RIV", 4: "ARN", 5: "OKC", 6: "NBG", 7: "WIN"},
    'warehouses': {i: wh for i, wh in enumerate([3112, 3115, 3139, 3070, 3005, 3095, 3083, 3145, 3204, 3187, 3156, 3177, 3198, 3125, 3195, 3167], 1)},
    'customers': {
        'Orlando Bubly/Gold': 10025, 'Pepsi STL Gold': 10043, 'AB STL 6 State Lids': 10049,
        'AB LA Stella': 10056, 'FCL Sealed Lids': 10059, 'AB STL Mango Cart': 10049,
        'AB STL Stella': 10049, 'AB JKSV Stella': 10052, 'AB FCL Shocktop': 10059,
        'AB STL Mango Cart 2': 10049, 'Elysian Hazi': 32222
    },
    'customer_warehouse_dict': {
        10025: 3005, 10029: 3115, 10033: 3125, 10036: 3112, 10038: 3005, 10040: 3177,
        10043: 3070, 10046: 3204, 10049: 3070, 10050: 3096, 10051: 3204, 10052: 3005,
        10054: 3041, 10055: 3095, 10056: 3112, 10057: 3115, 10058: 3200, 10059: 3139,
        10060: 3156, 10061: 3083, 10067: 3005, 13091: 3195, 17891: 3112, 17951: 3139,
        21281: 3112, 22671: 3198, 24171: 3156, 24172: 3167, 29746: 3112, 29751: 3115,
        29752: 3115, 29756: 3115, 30306: 3070, 30426: 3115, 30459: 3112, 31752: 3115,
        31770: 3070, 31876: 3115, 31966: 3005, 32222: 3112, 32631: 3115, 32632: 3112,
        32946: 3005, 33100: 3005, 33169: 3115, 33231: 3095, 33507: 3115, 33786: 3070,
        33861: 3083, 33906: 3112, 33918: 3139, 33919: 3112, 33931: 3145, 33933: 3145,
        33936: 3005, 33937: 3204, 33954: 3112, 34018: 3005, 34027: 3112, 34091: 3115,
        34127: 3112, 34217: 3145, 34221: 3005, 34222: 3112, 34223: 3112, 34321: 3005,
        34475: 3115, 34647: 3005, 34696: 3070, 34730: 3115, 34736: 3139, 34826: 3112,
        34946: 3145, 35011: 3145, 35116: 3112, 35161: 3070, 35176: 3070, 35177: 3115,
        35178: 3112, 35179: 3187, 35180: 3115, 35316: 3005, 35322: 3115, 35323: 3145,
        35361: 3115, 35451: 3145, 35512: 3115, 35513: 3187, 35522: 3005, 35537: 3115,
        35571: 3005, 35581: 3005, 35601: 3115, 35606: 3070, 35621: 3005, 35636: 3005,
        35697: 3115, 35707: 3070, 35753: 3112, 35755: 3070, 35756: 3115, 35757: 3167,
        35767: 3112, 35768: 3112, 35769: 3112, 35774: 3070, 35776: 3112, 35778: 3112,
        35782: 3115, 35784: 3112, 35785: 3115, 35787: 3115, 35788: 3112, 35791: 3139,
        35796: 3005, 35813: 3112, 35836: 3115, 35846: 3187, 35851: 3145, 35852: 3112,
        35862: 3005, 35908: 3145, 35912: 3187, 35916: 3112, 35917: 3005, 35937: 3115,
        35940: 3070, 35968: 3112, 35974: 3135, 35975: 3070, 35978: 3070, 35979: 3070,
        35980: 3070, 12375: 3112
    }
}

def format_millions(x, _):
    """Format y-axis numbers in millions with 1 decimal"""
    millions = x / 1e6
    if 0 < abs(millions) < 1:
        return f'${millions:.1f}M'
    else:
        return f'${int(millions)}M'

def format_number(x):
    """Format numbers with M/K suffix based on magnitude"""
    if abs(x) >= 1e3:
        # For values >= 1K show as integer thousands (e.g., 489K)
        thousands = x / 1e3
        return f'${int(round(thousands)):,}K'
    else:
        # For small values, show as integer
        return f'${int(round(x))}'
                
def create_progressive_total_cost_plots(data, month_order, output_dir):
    """
    Create three progressive total cost comparison plots:
    1. Current costs only
    2. Current + Base optimization
    3. Current + Base + Rules optimization
    """
    def add_value_labels(plt, x, y, color):
        """Add value labels above each point"""
        for i, v in enumerate(y):
            if pd.notnull(v) and v > 0:
                plt.text(x[i], v, format_number(v), 
                        ha='center', va='bottom',
                        rotation=0, fontsize=10,
                        color='k')
            
    colors = ['#4B0082', '#20B2AA', '#FF6B6B']  # Indigo, Light Sea Green, Coral

    # Plot configurations
    plots = [
        {
            'filename': 'total_cost_current.png',
            'title': 'Current Total Cost',
            'lines': [
                ('current_cost', 'Current Total', colors[0])
            ]
        },
        {
            'filename': 'total_cost_current_base.png',
            'title': 'Current vs Base Optimization Total Cost',
            'lines': [
                ('current_cost', 'Current Total', colors[0]),
                ('base_total_cost', 'Base Optimization', colors[1])
            ]
        },
        {
            'filename': 'total_cost_all.png',
            'title': 'Total Cost Comparison',
            'lines': [
                ('current_cost', 'Current Total', colors[0]),
                ('base_total_cost', 'Base Optimization', colors[1]),
                ('base_and_rules_total_cost', 'Base + Rules Optimization', colors[2])
            ]
        }
    ]
    
    # Create each plot
    for plot_config in plots:
        plt.figure(figsize=(12, 7))
        
        # Plot each line
        for column, label, color in plot_config['lines']:
            plt.plot(range(len(month_order)), data[column], '-o',
                    color=color, linewidth=1.5, label=label,
                    markersize=4)
            add_value_labels(plt, range(len(month_order)), data[column], color)
        
        plt.title(plot_config['title'], pad=20)
        plt.xlabel('Month')
        plt.ylabel('Total Cost')
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_millions))
        plt.xticks(range(len(month_order)), month_order, rotation=0)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f"{output_dir}/{plot_config['filename']}", 
                   dpi=300, bbox_inches='tight')
        plt.close()

def create_optimization_plots(df: pd.DataFrame, output_dir: str = './'):
    """
    Create optimization analysis plots from processed data.
    Generates four plots: product comparison, monthly savings analysis,
    shipping cost comparison, and total cost comparison.
    
    Args:
        df: Processed DataFrame from optimize_all_months
        output_dir: Directory to save plot files
    """
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    colors = ['#4B0082', '#20B2AA', '#FF6B6B']  # Indigo, Light Sea Green, Coral

    # 1. Savings Comparison by Product
    plt.figure(figsize=(12, 7))
    
    # Filter and prepare product data
    product_data = df[df['product_name'] != 'Total'].copy()
    product_base = product_data.groupby('product_name')['base_total_savings'].sum().fillna(0)
    product_rules = product_data.groupby('product_name')['base_and_rules_total_savings'].sum().fillna(0)
    
    # Filter out products with zero savings
    nonzero_mask = (product_base.round() > 0) | (product_rules.round() > 0)
    product_base = product_base[nonzero_mask]
    product_rules = product_rules[nonzero_mask]
    products = product_base.index
    
    x_prod = np.arange(len(products))
    width = 0.6
    
    # Create overlaid bars
    bars1 = plt.bar(x_prod, product_base, width,
                   label='Base Optimization',
                   color='#20B2AA',
                   alpha=0.4)
    bars2 = plt.bar(x_prod, product_rules, width,
                   label='Base + Rules Optimization',
                   color='#4169E1',
                   alpha=0.6)
    
    # Add value labels
    def add_value_labels(bars, values, offset):
        for bar, value in zip(bars, values):
            if value > 0:
                plt.text(bar.get_x() + bar.get_width()/2.,
                        value + offset,
                        format_number(value),
                        ha='center', va='bottom',
                        fontsize=10)
    
    add_value_labels(bars1, product_base, max(product_base)*0.03)
    add_value_labels(bars2, product_rules, max(product_rules)*0.01)
    
    plt.xlabel('Product')
    plt.ylabel('Savings')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_millions))
    plt.title('Savings Comparison by Product', pad=20)
    plt.xticks(x_prod, products, rotation=0)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save product comparison plot
    plt.savefig(f'{output_dir}/savings_comparison_by_product.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Monthly Savings Analysis
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Filter and prepare monthly data
    monthly_data = df[df['product_name'] == 'Total'].copy()
    monthly_data = monthly_data[monthly_data['month'].isin(month_order)]

    base_savings = monthly_data['base_total_savings']
    base_savings_percent = (base_savings / monthly_data['current_cost']) * 100
    rules_savings = monthly_data['base_and_rules_total_savings']
    rules_savings_percent = (rules_savings / monthly_data['current_cost']) * 100

    x = np.arange(len(month_order))
    width = 0.5

    # Create overlaid bars
    bars1 = ax1.bar(x, base_savings, width,
                    label='Base Savings',
                    color='#20B2AA',
                    alpha=0.4)
    bars2 = ax1.bar(x, rules_savings, width,
                    label='Base + Rules Savings',
                    color='#4169E1',
                    alpha=0.6)

    # Add bar value labels
    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            if value > 0:
                ax1.text(bar.get_x() + bar.get_width()/2,
                        value + value*0.01,
                        format_number(value),
                        ha='center', va='bottom',
                        fontsize=10,
                        rotation=0)

    add_value_labels(bars1, base_savings)
    add_value_labels(bars2, rules_savings)

    # Configure primary y-axis
    ax1.set_ylabel('Total Savings')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_millions))

    # Create secondary y-axis for percentages
    ax2 = ax1.twinx()

    # Plot percentage lines
    line1 = ax2.plot(x, base_savings_percent,
                    color='#FF6B6B',
                    linestyle='--',
                    linewidth=1.5,
                    marker='o',
                    markersize=4,
                    label='Base Savings % to Current Total')

    line2 = ax2.plot(x, rules_savings_percent,
                    color='#9370DB',
                    linewidth=1.5,
                    marker='o',
                    markersize=4,
                    label='Base + Rules Savings % to Current Total')

    # Add percentage labels
    def add_percent_labels(x_pos, percentages, color, offset):
        for i, pct in enumerate(percentages):
            if pd.notnull(pct):
                bbox_props = dict(boxstyle="round,pad=0.3",
                                fc="white",
                                ec="gray",
                                alpha=0.8)
                ax2.text(x_pos[i], pct + offset,
                        f'{pct:.1f}%',
                        ha='center',
                        va='bottom',
                        color=color,
                        fontsize=7,
                        bbox=bbox_props)

    add_percent_labels(x, base_savings_percent, '#FF6B6B', 0)
    add_percent_labels(x, rules_savings_percent, '#9370DB', 0)

    # Configure axes and labels
    plt.xticks(x, month_order, rotation=0)
    ax2.set_ylabel('Savings Percentage')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
              loc='upper right',
              fontsize='small')

    plt.title('Savings Comparison by Month', pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save monthly savings plot
    plt.savefig(f'{output_dir}/savings_comparison_by_month.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Cost Comparison Plots
    def create_cost_comparison_plot(data, cost_type, columns, labels):
        plt.figure(figsize=(12, 7))
        
        for i, (col, label) in enumerate(zip(columns, labels)):
            plt.plot(range(len(month_order)), data[col], '-o',
                    color=colors[i], linewidth=1.5, label=label,
                    markersize=4)
            
            # Add value labels
            for j, val in enumerate(data[col]):
                if pd.notnull(val) and val > 0:
                    plt.text(j, val, format_number(val),
                            ha='center', va='bottom',
                            rotation=0, fontsize=10)
        
        plt.title(f'{cost_type} Cost Comparison', pad=20)
        plt.xlabel('Month')
        plt.ylabel(f'{cost_type} Cost')
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_millions))
        plt.xticks(range(len(month_order)), month_order, rotation=0)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'{output_dir}/{cost_type.lower()}_cost_comparison.png',
                    dpi=300, bbox_inches='tight')
        plt.close()

    # Filter data for total costs
    total_data = df[df['product_name'] == 'Total']
    total_data = total_data[total_data['month'].isin(month_order)]
    
    # Create separate plots for Production and Shipping costs
    cost_types = {
        'Production': {
            'columns': ['current_production', 'base_production_cost', 'base_and_rules_production_cost'],
            'labels': ['Current Production', 'Base Optimization', 'Base + Rules Optimization']
        },
        'Shipping': {
            'columns': ['current_shipping', 'base_shipping_cost', 'base_and_rules_shipping_cost'],
            'labels': ['Current Shipping', 'Base Optimization', 'Base + Rules Optimization']
        }
    }
    
    for cost_type, config in cost_types.items():
        create_cost_comparison_plot(total_data, cost_type, config['columns'], config['labels'])
    
    # Create progressive total cost comparison plots
    create_progressive_total_cost_plots(total_data, month_order, output_dir)

def load_and_clean_data(file_path: str, sheet_name: str) -> pd.DataFrame:
    """
    Load and clean optimization data from Excel file.
    """
    columns = ['i', 'j', 'k', 'c^p_{ij}', 'c^l_{ijk}', 'D_{ik}', 'C_{ij}', 'x_{ijk}']
    
    df = pd.read_excel(file_path, sheet_name=sheet_name)[columns]
    
    # Clean data step by step
    df = (df.apply(pd.to_numeric, errors='coerce')
          .loc[lambda x: x['D_{ik}'] != 0]  # Filter non-zero demand
          .dropna(subset=['c^p_{ij}'])      # Drop rows with no production cost
          .assign(**{'c^l_{ijk}': lambda x: x['c^l_{ijk}'].fillna(99999)})  # Fill missing shipping costs
          .reset_index(drop=True))
    
    return df

def get_problem_dimensions(df: pd.DataFrame) -> None:
    st.markdown(f"""
    - **Total rows**: {len(df)}
    - **# of Products**: {df['i'].nunique()}
    - **# of Plants**: {df['j'].nunique()} 
    - **# of Warehouses**: {df['k'].nunique()}
    """)

def check_feasibility(df: pd.DataFrame, mappings: dict) -> tuple[pd.DataFrame, bool]:
    """
    Check and display feasibility for each product.
    Returns a tuple of (feasibility DataFrame, overall feasibility boolean)
    """
    feasibility_data = []
    all_feasible = True # Track overall feasibility
    
    for i in sorted(df['i'].unique()):
        # Get total demand - sum over unique (i,k) combinations
        demand = df[df['i'] == i].groupby('k')['D_{ik}'].first().sum()
        
        # Get total capacity - sum over unique (i,j) combinations 
        capacity = df[df['i'] == i].groupby('j')['C_{ij}'].first().sum()
        
        # Create feasibility status
        is_feasible = capacity >= demand
        all_feasible &= is_feasible # Update overall feasibility
        
        feasibility_data.append({
            'Product Name': mappings['products'][int(i)], # Use product name from mappings
            'Sales (Demand)': f"{demand:.1f}",
            'Production Capacity': f"{capacity:.1f}", 
            'Feasible': str(is_feasible)
        })
        
        # Display as DataFrame with colored text only in Feasible column
        feasibility_df = pd.DataFrame(feasibility_data)

    def color_feasible(val):
        color = 'green' if val == 'True' else 'red'
        return f'color: {color}'
    
    st.dataframe(feasibility_df.style.map(color_feasible, subset=['Feasible']))
    
    return feasibility_df, all_feasible

def create_pattern_visualizations(df: pd.DataFrame, mappings: dict) -> None:
    """
    Create and display production and shipping pattern visualizations 
    with mapped names and codes.
    """
    # Create production pattern
    prod_pattern = pd.DataFrame('□', 
                            index=[mappings['products'][int(i)] for i in sorted(df['i'].unique())],
                            columns=[mappings['plants'][int(j)] for j in sorted(df['j'].unique())],
                            dtype=str)

    for _, row in df.iterrows():
        if row['C_{ij}'] > 0:
            prod_pattern.loc[mappings['products'][int(row['i'])], 
                           mappings['plants'][int(row['j'])]] = '■'

    st.write("Production Capability Map (■ = can produce, □ = cannot produce)")
    st.dataframe(prod_pattern)

    # Create shipping pattern  
    ship_pattern = pd.DataFrame('□',
                            index=[mappings['products'][int(i)] for i in sorted(df['i'].unique())],
                            columns=[str(mappings['warehouses'][int(k)]) for k in sorted(df['k'].unique())],
                            dtype=str)

    for _, row in df.iterrows():
        if row['D_{ik}'] > 0:
            ship_pattern.loc[mappings['products'][int(row['i'])],
                           str(mappings['warehouses'][int(row['k'])])] = '■'

    st.write("Shipping Demand Map (■ = has demand, □ = no demand)") 
    st.dataframe(ship_pattern)

def adjust_capacity(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify capacity issues, adjust capacity to mimic specific inventory usage, 
    then proportionally where needed (so that the linear program is feasible), and display results.
    """
    df_adj = df.copy()
    adjustments = []

    # Only run the initial capacity adjustment if x_{ijk} column exists
    if 'x_{ijk}' in df.columns:
        # Identify instances where current shipping exceeds capacity (due to inventory using)
        df['x_{ij}'] = df.groupby(['i', 'j'])['x_{ijk}'].transform('sum')
        df_merged = df[['i', 'j', 'x_{ij}', 'C_{ij}']].drop_duplicates()
        df_merged['capacity_exceeded'] = df_merged['x_{ij}'] > df_merged['C_{ij}']
        capacity_issues = df_merged[df_merged['capacity_exceeded']]
        
        if not capacity_issues.empty:
            # Adjust capacity to mimic the usage of inventory
            for _, row in capacity_issues.iterrows():
                i, j, x_ij, C_ij = row['i'], row['j'], row['x_{ij}'], row['C_{ij}']
                df_adj.loc[(df_adj['i'] == i) & (df_adj['j'] == j), 'C_{ij}'] = x_ij
                adjustments.append({'Product': int(i), 'Plant': int(j), 
                                    'Original': C_ij, 'Adjusted': x_ij, 
                                    'Scale': x_ij / C_ij})
    
    # Proportionally adjust remaining capacity if needed so that the linear program is feasible
    for i in df['i'].unique():
        total_demand = df[df['i'] == i].groupby('k')['D_{ik}'].first().sum()
        total_capacity = df_adj[df_adj['i'] == i].groupby('j')['C_{ij}'].first().sum()
        
        if total_capacity < total_demand:
            alpha = total_demand / total_capacity
            mask = (df['i'] == i) & (~df.index.isin(capacity_issues.index))
            
            for j in df[mask]['j'].unique():
                original = df_adj.loc[(df_adj['i'] == i) & (df_adj['j'] == j), 'C_{ij}'].iloc[0] 
                df_adj.loc[(df_adj['i'] == i) & (df_adj['j'] == j), 'C_{ij}'] *= alpha
                adjusted = df_adj.loc[(df_adj['i'] == i) & (df_adj['j'] == j), 'C_{ij}'].iloc[0]
                adjustments.append({'Product': int(i), 'Plant': int(j),
                                    'Original': original, 'Adjusted': adjusted,  
                                    'Scale': alpha})
        
    return df_adj, pd.DataFrame(adjustments)

def get_sourcing_constraints(df, x, sheet_name):
    """Define sourcing constraints based on month."""
    # Basic minimum requirements in tuple form (name, i, j, k, min_val)
    basic_reqs = [
        ("Orlando Bubly/Gold", 4, 3, 5, 10),
        ("STL Gold", 4, 3, 4, 1),
        ("STL 6 State Lids", 4, 3, 4, 1),
        ("LA Stella", 4, 5, 1, 3),
        ("AB STL Mango Cart", 1, 6, 4, 1.2),
        ("AB STL Stella", 1, 2, 4, 0.15),
        ("AB LA Stella", 3, 6, 1, 0.05),
        ("AB JKSV Stella", 3, 6, 5, 0.01),
        ("AB FCL Shocktop", 3, 2, 3, 0.075),
        ("AB STL Mango Cart 2", 3, 2, 4, 0.155),
        ("Elysian Hazi", 3, 2, 1, 0.02)
    ]
    
    # Convert basic requirements to constraint format
    constraints = [
        {
            "name": name,
            "indices": [(i, j, k)],
            "constraints": (x[i,j,k] >= min_val,)
        }
        for name, i, j, k, min_val in basic_reqs
    ]
    
    # Add FCL Sealed Lids split requirement
    fcl_demand = df.loc[(df['i'] == 4) & (df['k'] == 3), 'D_{ik}'].values[0]
    constraints.extend([
        {
            "name": "FCL Sealed Lids - 20% Plant 5",
            "indices": [(4,5,3)],
            "constraints": (x[4,5,3] >= fcl_demand * 0.2,)
        },
        {
            "name": "FCL Sealed Lids - 80% Plant 3",
            "indices": [(4,3,3)],
            "constraints": (x[4,3,3] >= fcl_demand * 0.8,)
        }
    ])
    
    # Add seasonal constraint if applicable
    if sheet_name.split('-')[1].lower() in ['sep', 'sept', 'september', 'oct', 'october', 'nov', 'november', 'dec', 'december']:
        constraints.append({
            "name": "AB HOU OLOF Lids",
            "indices": [(4,3,6)],
            "constraints": (x[4,3,6] >= 25,)
        })
    
    return constraints

def test_and_apply_constraints(df, x, constraints, base_constraints, objective):
    """Test and apply sourcing constraints, printing results."""
    working_constraints = base_constraints.copy()
    
    for c in constraints:
        try:
            test_constraints = working_constraints + list(c["constraints"])
            problem = cp.Problem(objective, test_constraints)
            problem.solve()
            
            if problem.status == 'optimal':
                working_constraints.extend(c["constraints"])
            else:
                print(f"\nINFEASIBLE CONSTRAINT FOUND: {c['name']}")
                for i, j, k in c["indices"]:
                    demand = df.loc[(df['i'] == i) & (df['k'] == k), 'D_{ik}'].values[0]
                    capacity = df.loc[(df['i'] == i) & (df['j'] == j), 'C_{ij}'].values[0]
                    print(f"Product {i}, Plant {j}, Warehouse {k}:")
                    print(f"Demand: {demand}, Capacity: {capacity}")
        except Exception as e:
            print(f"Error with {c['name']}: {e}\n")
    
    return problem, working_constraints

def optimize_month(df, sheet_name):
    """Run optimization for a single month"""
    # Define variables
    x = {(i, j, k): cp.Variable(nonneg=True) 
         for i, j, k in zip(df['i'], df['j'], df['k'])}
    
    # Define objective
    objective = cp.Minimize(cp.sum(
        [(df.loc[idx, 'c^p_{ij}'] + df.loc[idx, 'c^l_{ijk}']) * x[i, j, k] 
         for idx, (i, j, k) in enumerate(zip(df['i'], df['j'], df['k']))]
    ))
    
    # Base constraints
    base_constraints = []
    
    # Add demand constraints
    for (i, k), group in df.groupby(['i', 'k']):
        base_constraints.append(
            cp.sum([x[i, j, k] for j in group['j']]) >= group['D_{ik}'].values[0]
        )
    
    # Add capacity constraints
    for (i, j), group in df.groupby(['i', 'j']):
        base_constraints.append(
            cp.sum([x[i, j, k] for k in group['k']]) <= group['C_{ij}'].values[0]
        )
    
    # Solve with base constraints
    base_problem = cp.Problem(objective, base_constraints)
    base_problem.solve()
    
    # Store base optimization results
    df['x^*_{ijk}_base'] = [float(x[i, j, k].value) for i, j, k in zip(df['i'], df['j'], df['k'])]
    df['base_total_cost'] = (df['c^p_{ij}'] + df['c^l_{ijk}']) * df['x^*_{ijk}_base']
    df['base_production_cost'] = df['c^p_{ij}'] * df['x^*_{ijk}_base']
    df['base_shipping_cost'] = df['c^l_{ijk}'] * df['x^*_{ijk}_base']
    df['diff_base'] = df['x^*_{ijk}_base'] - df['x_{ijk}']
    
    # Get additional sourcing constraints
    sourcing_constraints = get_sourcing_constraints(df, x, sheet_name)
    final_problem, final_constraints = test_and_apply_constraints(
        df, x, sourcing_constraints, base_constraints, objective
    )
    
    # Store additional optimization results
    df['x^*_{ijk}_base_and_rules'] = [float(x[i, j, k].value) for i, j, k in zip(df['i'], df['j'], df['k'])]
    df['base_and_rules_total_cost'] = (df['c^p_{ij}'] + df['c^l_{ijk}']) * df['x^*_{ijk}_base_and_rules']
    df['base_and_rules_production_cost'] = df['c^p_{ij}'] * df['x^*_{ijk}_base_and_rules']
    df['base_and_rules_shipping_cost'] = df['c^l_{ijk}'] * df['x^*_{ijk}_base_and_rules']
    df['diff_base_and_rules'] = df['x^*_{ijk}_base_and_rules'] - df['x_{ijk}']

    return df

def optimize_all_months(uploaded_file, mappings):
    """Process all months and return key DataFrames for analysis"""
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    summaries = []
    monthly_data = []
    
    # Process each month
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, month in enumerate(month_order):
        status_text.text(f"Processing {month}...")
        
        try:
            # Load and clean data for the current month
            sheet_name = f'Solver-{month}'
            df = load_and_clean_data(uploaded_file, sheet_name)
            
            # Display problem dimensions
            with st.expander(f"Problem Details - {month}", expanded=False):
                st.write("#### Problem Dimensions")
                get_problem_dimensions(df)
                
                st.write("#### Initial Feasibility Check")
                feasibility_df, all_feasible = check_feasibility(df, mappings)
                
                st.write("#### Production and Shipping Patterns")
                create_pattern_visualizations(df, mappings)

                # Adjust capacity if needed
                df_adjusted, adjustments = adjust_capacity(df)
                if not adjustments.empty:
                    if not all_feasible:
                        st.write("#### Capacity Adjustments")
                        st.dataframe(adjustments)
                        st.write("#### Final Feasibility Check")
                        check_feasibility(df_adjusted, mappings)
                    df = df_adjusted
            
            # Add plant_code, warehouse_id, and customer_id columns
            df['product_name'] = df['i'].map(mappings['products'])
            df['plant_code'] = df['j'].map(mappings['plants'])
            df['warehouse_id'] = df['k'].map(mappings['warehouses'])

            # Run optimization for the current month
            df = optimize_month(df, sheet_name)
            
            # Calculate costs
            df['product_name'] = df['i'].map(mappings['products'])
            
            # Current costs
            df['current_cost'] = (df['c^p_{ij}'] + df['c^l_{ijk}']) * df['x_{ijk}']
            df['current_production'] = df['c^p_{ij}'] * df['x_{ijk}']
            df['current_shipping'] = df['c^l_{ijk}'] * df['x_{ijk}']
            
            # Store monthly data
            monthly_data.append({'month': month, 'data': df})
            
            # Create product summary
            cols_to_sum = ['current_cost', 'current_production', 'current_shipping',
                          'base_total_cost', 'base_production_cost', 'base_shipping_cost',
                          'base_and_rules_total_cost', 'base_and_rules_production_cost', 'base_and_rules_shipping_cost']
            prod_sum = df.groupby('product_name')[cols_to_sum].sum().reset_index()
            prod_sum['month'] = month
            
            # Create total summary using the actual cost calculation
            total_current = {
                'total': sum((row['c^p_{ij}'] + row['c^l_{ijk}']) * row['x_{ijk}'] for _, row in df.iterrows()),
                'production': sum(row['c^p_{ij}'] * row['x_{ijk}'] for _, row in df.iterrows()),
                'shipping': sum(row['c^l_{ijk}'] * row['x_{ijk}'] for _, row in df.iterrows())
            }
            
            total_base = {
                'total': sum((row['c^p_{ij}'] + row['c^l_{ijk}']) * row['x^*_{ijk}_base'] for _, row in df.iterrows()),
                'production': sum(row['c^p_{ij}'] * row['x^*_{ijk}_base'] for _, row in df.iterrows()),
                'shipping': sum(row['c^l_{ijk}'] * row['x^*_{ijk}_base'] for _, row in df.iterrows())
            }
            
            total_rules = {
                'total': sum((row['c^p_{ij}'] + row['c^l_{ijk}']) * row['x^*_{ijk}_base_and_rules'] for _, row in df.iterrows()),
                'production': sum(row['c^p_{ij}'] * row['x^*_{ijk}_base_and_rules'] for _, row in df.iterrows()),
                'shipping': sum(row['c^l_{ijk}'] * row['x^*_{ijk}_base_and_rules'] for _, row in df.iterrows())
            }
            
            total = pd.DataFrame([{
                'product_name': 'Total',
                'month': month,
                'current_cost': total_current['total'],
                'current_production': total_current['production'],
                'current_shipping': total_current['shipping'],
                'base_total_cost': total_base['total'],
                'base_production_cost': total_base['production'],
                'base_shipping_cost': total_base['shipping'],
                'base_and_rules_total_cost': total_rules['total'],
                'base_and_rules_production_cost': total_rules['production'],
                'base_and_rules_shipping_cost': total_rules['shipping']
            }])
            
            # Calculate savings
            for df_temp in [prod_sum, total]:
                # Total cost savings
                df_temp['base_total_savings'] = df_temp['current_cost'] - df_temp['base_total_cost']
                df_temp['base_and_rules_total_savings'] = df_temp['current_cost'] - df_temp['base_and_rules_total_cost']
                
                # Production cost savings
                df_temp['base_production_savings'] = df_temp['current_production'] - df_temp['base_production_cost']
                df_temp['base_and_rules_production_savings'] = df_temp['current_production'] - df_temp['base_and_rules_production_cost']
                
                # Shipping cost savings
                df_temp['base_shipping_savings'] = df_temp['current_shipping'] - df_temp['base_shipping_cost']
                df_temp['base_and_rules_shipping_savings'] = df_temp['current_shipping'] - df_temp['base_and_rules_shipping_cost']
            
            summaries.append({
                'summary': pd.concat([prod_sum, total])
            })
            
        except Exception as e:
            st.error(f"Error processing {month}: {e}")
            return None, None, None
        
        # Update progress
        progress_bar.progress((i + 1) / len(month_order))
    
    if summaries:
        # Combine all summaries
        combined = pd.concat([data['summary'] for data in summaries])
        
        # Calculate annual summary
        annual = combined[combined['product_name'] == 'Total']
        annual_sum = pd.DataFrame([{
            'month': 'Annual Total',
            'product_name': 'Total',
            **{col: annual[col].sum() for col in annual.columns 
               if col not in ['month', 'product_name']}
        }])
        
        # Create final DataFrame with annual summary
        results_df = pd.concat([combined, annual_sum], ignore_index=True)
        
        # Create percentage summary
        percentage_summary = combined[combined['product_name'] == 'Total'].copy()
        percentage_cols = {
            'base_total_savings': 'Base Total',
            'base_production_savings': 'Base Production',
            'base_shipping_savings': 'Base Shipping',
            'base_and_rules_total_savings': 'Base+Rules Total',
            'base_and_rules_production_savings': 'Base+Rules Production',
            'base_and_rules_shipping_savings': 'Base+Rules Shipping'
        }
        
        for col, name in percentage_cols.items():
            percentage_summary[f'{name} %'] = (
                percentage_summary[col] / percentage_summary['current_cost'] * 100
            )
        
        status_text.text("Processing complete!")
        return {
            'monthly_data': monthly_data,
            'results_df': results_df,
            'percentage_summary': percentage_summary,
            'annual_summary': annual_sum
        }
    
    return None

def create_supply_chain_map(plants, warehouses):
    """Creates and displays an interactive supply chain network map in Jupyter"""
    # Create map and add style
    m = folium.Map(location=[39.8, -91], zoom_start=4)
    
    # Custom popup style
    popup_style = """
    <style>
        .custom-popup .leaflet-popup-content-wrapper {
            background: rgba(255, 255, 255, 0.95);
            color: #333;
            border-radius: 5px;
            padding: 1px;
            width: auto !important;
            min-width: 200px;
        }
        .custom-popup .leaflet-popup-content {
            margin: 10px;
            line-height: 1.4;
        }
        .custom-popup .leaflet-popup-tip {
            background: rgba(255, 255, 255, 0.95);
        }
    </style>
    """
    m.get_root().header.add_child(folium.Element(popup_style))

    # Add plant markers with 'industry' icon
    for _, p in plants.iterrows():
        if pd.notna(p['lat']):
            popup_html = f"""<b>Plant: {p['Plant #']}</b><br>
                Code: {p['Code']}, {p['City']}, {p['State']}<br>
                Type: {p['Type']} Plant"""
            icon_color = 'blue' if p['Type'] == 'Can' else 'red'
            folium.Marker(
                location=[p['lat'] + np.random.uniform(-0.005, 0.005),  
                          p['lon'] + np.random.uniform(-0.005, 0.005)], # Add random noise to avoid display overlap 
                icon=folium.Icon(color=icon_color, icon='industry', prefix='fa'),
                popup=folium.Popup(popup_html, max_width=300),
                # opacity=0.9
            ).add_to(m)

    # Add warehouse markers with 'warehouse' icon
    for _, w in warehouses.iterrows():
        if pd.notna(w['lat']):
            popup_html = f"""<b>Warehouse: {w['ID']}</b><br>
                Name: {w['Name']}, {w['City']}, {w['State']}"""
            folium.Marker(
                location=[w['lat'] + np.random.uniform(-0.005, 0.005), 
                          w['lon'] + np.random.uniform(-0.005, 0.005)], # Add random noise to avoid display overlap 
                icon=folium.Icon(color='green', icon='warehouse', prefix='fa'),
                popup=folium.Popup(popup_html, max_width=300),
                opacity=0.7
            ).add_to(m)

    legend_html = """
        <div style="position: fixed; 
                    top: 15px; left: 50px;
                    max-width: 90%; /* Adjust width to be responsive */
                    border: 2px solid grey; 
                    z-index: 1000;
                    background-color: white;
                    padding: calc(0.5vw); /* Responsive padding */
                    font-family: Arial;
                    font-size: calc(0.8vw); /* Responsive font size */
                    opacity: 0.9;
                    box-sizing: border-box;">
            <div style="margin-bottom: 5px"><strong>Markers</strong></div>
            <div style="margin-bottom: 3px">
                <i class="fa fa-industry" style="color:#39a9dc"></i> Can Plant
            </div>
            <div style="margin-bottom: 3px">
                <i class="fa fa-industry" style="color:#df3822"></i> Lid Plant
            </div>
            <div style="margin-bottom: 3px">
                <i class="fa fa-warehouse" style="color:#72b024"></i> Warehouse
            </div>
        </div>
        """

    tables_html = f"""
        <div id="table-container" style="
                    position: fixed; 
                    top: 15px; right: 15px;
                    max-width: 90%; 
                    border: 2px solid grey; 
                    z-index: 9999; 
                    background-color: white;
                    padding: 6px;
                    font-family: Arial;
                    font-size: 12px;
                    opacity: 0.9;
                    overflow: auto; 
                    box-sizing: border-box;
                    max-height: calc(100vh - 30px); /* Dynamically adjusts to viewport height */
                    overflow-y: auto;">
            <div style="margin-bottom: 3px"><strong>Manufacturing Plants</strong></div>
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 8px">
                <tr style="background-color: #f0f0f0">
                    <th style="border: 1px solid #ddd; padding: 2px">Plant #</th>
                    <th style="border: 1px solid #ddd; padding: 2px">Code</th>
                    <th style="border: 1px solid #ddd; padding: 2px">Location</th>
                    <th style="border: 1px solid #ddd; padding: 2px">Type</th>
                </tr>
                {''.join(f"<tr><td style='border: 1px solid #ddd; padding: 2px'>{row['Plant #']}</td>"
                        f"<td style='border: 1px solid #ddd; padding: 2px'>{row['Code']}</td>"
                        f"<td style='border: 1px solid #ddd; padding: 2px'>{row['City']}, {row['State']}</td>"
                        f"<td style='border: 1px solid #ddd; padding: 2px'>{row['Type']}</td></tr>" 
                        for _, row in plants.iterrows())}
            </table>
            <div style="margin-bottom: 3px"><strong>Warehouses</strong></div>
            <table style="width: 100%; border-collapse: collapse">
                <tr style="background-color: #f0f0f0">
                    <th style="border: 1px solid #ddd; padding: 2px">Warehouse #</th>
                    <th style="border: 1px solid #ddd; padding: 2px">Name</th>
                    <th style="border: 1px solid #ddd; padding: 2px">Location</th>
                </tr>
                {''.join(f"<tr><td style='border: 1px solid #ddd; padding: 2px'>{row['ID']}</td>"
                        f"<td style='border: 1px solid #ddd; padding: 2px'>{row['Name']}</td>"
                        f"<td style='border: 1px solid #ddd; padding: 2px'>{row['City']}, {row['State']}</td></tr>" 
                        for _, row in warehouses.iterrows())}
            </table>
        </div>
    """

    resize_script = """
        <script>
            document.addEventListener('DOMContentLoaded', function () {
                function adjustTableHeight() {
                    const mapHeight = document.querySelector('.leaflet-container')?.clientHeight || window.innerHeight;
                    const tableContainer = document.getElementById('table-container');
                    if (tableContainer) {
                        tableContainer.style.maxHeight = (mapHeight - 30) + 'px';
                    }
                }
                window.addEventListener('resize', adjustTableHeight);
                adjustTableHeight();
            });
        </script>
    """

    # Add legend, table, and resize script to the map
    m.get_root().html.add_child(folium.Element(legend_html))
    m.get_root().html.add_child(folium.Element(tables_html))
    return m

def create_shipping_routes_map(df_plants, df_warehouses, monthly_data, product_id=4, base_only=True, month='Jan', mappings=mappings):
    """
    Create an interactive map showing shipping route changes with animated paths using AntPath.
    Markers for plants and warehouses are displayed only if they are involved in the routes.
    Adds numeric labels above the middle of each arrow.
    """
    # Get data for the specified month
    month_data = next((data['data'] for data in monthly_data if data['month'] == month), None)
    if month_data is None:
        raise ValueError(f"No data found for month: {month}")
    
    # Filter for the specific product
    df_product = month_data[month_data['i'] == product_id].copy()
    
    # Get costs directly from df_product
    current_ship_cost = df_product['current_shipping'].sum()
    current_prod_cost = df_product['current_production'].sum()
    current_total = df_product['current_cost'].sum()
    
    if base_only:
        opt_ship_cost = df_product['base_shipping_cost'].sum()
        opt_prod_cost = df_product['base_production_cost'].sum()
        opt_total = df_product['base_total_cost'].sum()
    else:
        opt_ship_cost = df_product['base_and_rules_shipping_cost'].sum()
        opt_prod_cost = df_product['base_and_rules_production_cost'].sum()
        opt_total = df_product['base_and_rules_total_cost'].sum()
    
    # Filter for non-zero differences
    diff_col = 'diff_base' if base_only else 'diff_base_and_rules'
    routes_df = df_product[abs(df_product[diff_col]) > 1e-5].copy()
    routes_df = routes_df.sort_values(diff_col, key=abs, ascending=False)
    
    # Ensure all IDs are strings
    routes_df['plant_code'] = routes_df['plant_code'].astype(str)
    routes_df['warehouse_id'] = routes_df['warehouse_id'].astype(str)
    df_plants['Code'] = df_plants['Code'].astype(str)
    df_warehouses['ID'] = df_warehouses['ID'].astype(str)

    # Add random noise to coordinates to avoid overlap cases
    df_plants['lat'] += np.random.uniform(-0.005, 0.005, size=len(df_plants))
    df_plants['lon'] += np.random.uniform(-0.005, 0.005, size=len(df_plants))
    df_warehouses['lat'] += np.random.uniform(-0.005, 0.005, size=len(df_warehouses))
    df_warehouses['lon'] += np.random.uniform(-0.005, 0.005, size=len(df_warehouses))
    
    # Create reverse lookup for warehouse to customers
    warehouse_customers_dict = defaultdict(list)
    for customer, warehouse in mappings['customer_warehouse_dict'].items():
        warehouse_customers_dict[str(warehouse)].append(customer)
    
    # Add warehouse names and associated customers to routes_df
    routes_df['warehouse_name'] = routes_df['warehouse_id'].map(df_warehouses.set_index('ID')['Name'].get)
    routes_df['associated_customers'] = routes_df['warehouse_id'].map(lambda x: ', '.join(map(str, warehouse_customers_dict[x])))
    
    # Prepare mappings for coordinates
    plant_coords = dict(zip(df_plants['Code'], zip(df_plants['lat'], df_plants['lon'])))
    warehouse_coords = dict(zip(df_warehouses['ID'], zip(df_warehouses['lat'], df_warehouses['lon'])))
    
    # Create the map centered on the US
    m = folium.Map(location=[30, -98.5795], zoom_start=5)
    
    # Add markers only for relevant plants
    for _, row in df_plants[df_plants['Code'].isin(routes_df['plant_code'].unique())].iterrows():
        popup_html = f"""<b>Plant: {row['Plant #']}</b><br>
                Code: {row['Code']}, {row['City']}, {row['State']}<br>
                Type: {row['Type']} Plant"""
        icon_color = 'blue' if row['Type'] == 'Can' else 'red'
        folium.Marker(
            location=[row['lat'], row['lon']],
            icon=folium.Icon(color=icon_color, icon='industry', prefix='fa'),
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(m)
    
    # Add markers only for relevant warehouses
    for _, row in df_warehouses[df_warehouses['ID'].isin(routes_df['warehouse_id'].unique())].iterrows():
        popup_html = f"""<b>Warehouse: {row['ID']}</b><br>
                    Name: {row['Name']}, {row['City']}, {row['State']}<br>
                    Customers: {routes_df[routes_df['warehouse_id'] == row['ID']]['associated_customers'].values[0]}"""
        folium.Marker(
            location=[row['lat'], row['lon']],
            icon=folium.Icon(color='green', icon='warehouse', prefix='fa'),
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(m)

    # Add animated routes with AntPath and numeric labels
    for _, row in routes_df.iterrows():
        plant_code = row['plant_code']
        warehouse_id = row['warehouse_id']
        value = row[diff_col]
        if plant_code in plant_coords and warehouse_id in warehouse_coords:
            plant_loc = plant_coords[plant_code]
            wh_loc = warehouse_coords[warehouse_id]
            
            # Add AntPath for animation
            folium.plugins.AntPath(
                locations=[plant_loc, wh_loc],
                color='green' if value > 0 else 'red',  # Blue for increase, red for decrease
                weight=2 + abs(value) / 20,  # Scale weight by magnitude
                opacity=0.7,
                dash_array=[10, 20],  # Dashed line style
                delay=1000  # Animation delay
            ).add_to(m)
            
            # Add numeric label at the midpoint
            folium.Marker(
                location=[(plant_loc[0] + wh_loc[0]) / 2, (plant_loc[1] + wh_loc[1]) / 2],
                icon=folium.DivIcon(
                    icon_size=(150, 24),
                    icon_anchor=(0, 0),
                    html=f'<div style="font-size: 12px; font-weight: bold; color: {"green" if value > 0 else "red"};">{"+" if value > 0 else ""}{value:.2f}</div>'
                )
            ).add_to(m)
        else:
            print(f"Missing coordinates for plant {plant_code} or warehouse {warehouse_id}")
    
    # Add legend for cost summary
    legend_html = f"""
        <div style="position: fixed; 
                    top: 15px; right: 15px; 
                    border: 2px solid grey; z-index: 9999; 
                    background-color: white;
                    padding: calc(0.5vw); /* Responsive padding */
                    font-family: Arial;
                    font-size: calc(0.9vw); /* Responsive font size */
                    opacity: 0.9;
                    max-width: 90%; /* Prevents overflow */
                    box-sizing: border-box;">
        <p style="margin: 0; font-weight: bold;">Suggested Shipping Routes</p>
        <p style="margin: 0;">Product: {mappings['products'].get(product_id, f'Product {product_id}')}</p>
        <p style="margin: 0;">Optimization: {'Base Only' if base_only else 'Base + Rules'}</p>
        <p style="margin: 0; color: green;">Green: Increased Volume →</p>
        <p style="margin: 0; color: red;">Red: Decreased Volume →</p>
        <p style="margin: 0;">Line thickness indicates magnitude</p>
        <hr style="margin: 5px 0;">
        <p style="margin: 0; font-weight: bold;">Current Total Cost: ${current_total:,.2f}</p>
        <p style="margin: 0;">Shipping: ${current_ship_cost:,.2f}</p>
        <p style="margin: 0;">Production: ${current_prod_cost:,.2f}</p>
        <hr style="margin: 5px 0;">
        <p style="margin: 0; font-weight: bold;">Optimized Total Cost: ${opt_total:,.2f}</p>
        <p style="margin: 0;">Shipping: ${opt_ship_cost:,.2f}</p>
        <p style="margin: 0;">Production: ${opt_prod_cost:,.2f}</p>
        <hr style="margin: 5px 0;">
        <p style="margin: 0; font-weight: bold;">Total Savings: ${current_total - opt_total:,.2f}</p>
        </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m, routes_df

def get_supply_chain_locations():
    """Retrieves and returns supply chain location data with coordinates"""
    
    # Data setup
    plants = pd.DataFrame([
        ['5101', 'JAX', 'Jacksonville', 'FL', 'Can'],
        ['5103', 'ARN', 'Arnold', 'MO', 'Can'],
        ['5107', 'WIN', 'Windsor', 'CO', 'Can'],
        ['5109', 'NBG', 'Newburgh', 'NY', 'Can'],
        ['5112', 'MIR', 'Mira Loma', 'CA', 'Can'],
        ['5205', 'OKC', 'Oklahoma City', 'OK', 'Lid'],
        ['5208', 'RIV', 'Riverside', 'CA', 'Lid']
    ], columns=['Plant #', 'Code', 'City', 'State', 'Type'])

    warehouses = pd.DataFrame([
        ['3005', 'Biagi Jax 2', 'Jacksonville', 'FL'],
        ['3041', 'Liberty Williamsburg', 'Williamsburg', 'VA'],
        ['3070', 'Stitch-Tec 1st Street', 'St. Louis', 'MO'],
        ['3078', 'Stitch-Tech 23rd Street', 'St. Louis', 'MO'],
        ['3083', 'Gateway', 'Cartersville', 'GA'],
        ['3095', 'Buske Houston', 'Houston', 'TX'],
        ['3096', 'Ainsley', 'Baldwinsville', 'NY'],
        ['3112', 'Biagi Ontario', 'Ontario', 'CA'],
        ['3115', 'NFI Newburgh', 'Newburgh', 'NY'],
        ['3125', 'Updike Woodland', 'Woodland', 'CA'],
        ['3138', 'Quarterback Warehouse', 'Cambridge', 'ON'],
        ['3139', 'TMSI-Windsor', 'Windsor', 'CO'],
        ['3145', 'Biagi OKC', 'Oklahoma City', 'OK'],
        ['3156', 'Updike Vacaville', 'Vacaville', 'CA'],
        ['3167', 'Biagi Auburn', 'Auburn', 'WA'],
        ['3187', 'Biagi Jax 3', 'Jacksonville', 'FL'],
        ['3190', 'STC Nashville', 'Nashville', 'IL'],
        ['3195', 'United Tulsa', 'Tulsa', 'OK'],
        ['3200', 'NFI Columbus', 'Columbus', 'OH'],
        ['3201', 'NFI Port Reading', 'Port Reading', 'NJ'],
        ['3202', 'Biagi Van Nuys', 'Van Nuys', 'CA'],
        ['3204', 'NFI Distribution', 'Edison', 'NJ']
    ], columns=['ID', 'Name', 'City', 'State'])

    def get_coords(city, state):
        try:
            loc = Nominatim(user_agent="supply_chain").geocode(f"{city}, {state}, {'Canada' if state=='ON' else 'USA'}")
            return (loc.latitude, loc.longitude) if loc else (None, None)
        except GeocoderTimedOut: return None, None

    # Get coordinates
    plants[['lat','lon']] = plants.apply(lambda x: pd.Series(get_coords(x['City'], x['State'])), axis=1)
    warehouses[['lat','lon']] = warehouses.apply(lambda x: pd.Series(get_coords(x['City'], x['State'])), axis=1)
    
    print(f"Plants mapped: {len(plants[plants.lat.notna()])} of {len(plants)}")
    print(f"Warehouses mapped: {len(warehouses[warehouses.lat.notna()])} of {len(warehouses)}")
    
    return plants, warehouses

# df_plants, df_warehouses = get_supply_chain_locations()

# Pre-defined plants data with coordinates
df_plants = pd.DataFrame([
    ['5101', 'JAX', 'Jacksonville', 'FL', 'Can', 30.332184, -81.655651],
    ['5103', 'ARN', 'Arnold', 'MO', 'Can', 38.422671, -90.375829],
    ['5107', 'WIN', 'Windsor', 'CO', 'Can', 40.477482, -104.901361],
    ['5109', 'NBG', 'Newburgh', 'NY', 'Can', 41.503427, -74.010418],
    ['5112', 'MIR', 'Mira Loma', 'CA', 'Can', 33.986391, -117.522733],
    ['5205', 'OKC', 'Oklahoma City', 'OK', 'Lid', 35.472989, -97.517054],
    ['5208', 'RIV', 'Riverside', 'CA', 'Lid', 33.982495, -117.374238]
], columns=['Plant #', 'Code', 'City', 'State', 'Type', 'lat', 'lon'])

# Pre-defined warehouses data with coordinates
df_warehouses = pd.DataFrame([
    ['3005', 'Biagi Jax 2', 'Jacksonville', 'FL', 30.332184, -81.655651],
    ['3041', 'Liberty Williamsburg', 'Williamsburg', 'VA', 37.270879, -76.707404],
    ['3070', 'Stitch-Tec 1st Street', 'St. Louis', 'MO', 38.628028, -90.191015],
    ['3078', 'Stitch-Tech 23rd Street', 'St. Louis', 'MO', 38.628028, -90.191015],
    ['3083', 'Gateway', 'Cartersville', 'GA', 34.165230, -84.799761],
    ['3095', 'Buske Houston', 'Houston', 'TX', 29.758938, -95.367697],
    ['3096', 'Ainsley', 'Baldwinsville', 'NY', 43.158679, -76.332710],
    ['3112', 'Biagi Ontario', 'Ontario', 'CA', 34.065846, -117.648430],
    ['3115', 'NFI Newburgh', 'Newburgh', 'NY', 41.503427, -74.010418],
    ['3125', 'Updike Woodland', 'Woodland', 'CA', 38.678611, -121.773329],
    ['3138', 'Quarterback Warehouse', 'Cambridge', 'ON', 43.360054, -80.312302],
    ['3139', 'TMSI-Windsor', 'Windsor', 'CO', 40.477482, -104.901361],
    ['3145', 'Biagi OKC', 'Oklahoma City', 'OK', 35.472989, -97.517054],
    ['3156', 'Updike Vacaville', 'Vacaville', 'CA', 38.356577, -121.987744],
    ['3167', 'Biagi Auburn', 'Auburn', 'WA', 47.307537, -122.230181],
    ['3187', 'Biagi Jax 3', 'Jacksonville', 'FL', 30.332184, -81.655651],
    ['3190', 'STC Nashville', 'Nashville', 'IL', 38.521441, -89.380075],
    ['3195', 'United Tulsa', 'Tulsa', 'OK', 36.153982, -95.992775],
    ['3200', 'NFI Columbus', 'Columbus', 'OH', 39.961176, -82.998794],
    ['3201', 'NFI Port Reading', 'Port Reading', 'NJ', 40.564270, -74.264641],
    ['3202', 'Biagi Van Nuys', 'Van Nuys', 'CA', 34.189857, -118.451357],
    ['3204', 'NFI Distribution', 'Edison', 'NJ', 40.518716, -74.412095]
], columns=['ID', 'Name', 'City', 'State', 'lat', 'lon'])

def validate_uploaded_file(uploaded_file):
    """
    Validate the uploaded Excel file for:
    1. Existence of required sheets for all months.
    2. Presence of required columns in each sheet.
    
    Args:
        uploaded_file: The uploaded Excel file.
        
    Returns:
        bool: True if the file is valid, False otherwise.
        str: Error message if invalid, empty string otherwise.
    """
    required_columns = ['c^p_{ij}', 'c^l_{ijk}', 'D_{ik}', 'C_{ij}']
    required_sheets = [f'Solver-{month}' for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]
    
    try:
        # Load the Excel file
        excel_data = pd.ExcelFile(uploaded_file)
        missing_sheets = [sheet for sheet in required_sheets if sheet not in excel_data.sheet_names]
        
        if missing_sheets:
            return False, f"Missing required sheets: {', '.join(missing_sheets)}"
        
        # Check for required columns in each sheet
        for sheet in required_sheets:
            sheet_data = pd.read_excel(excel_data, sheet_name=sheet)
            missing_columns = [col for col in required_columns if col not in sheet_data.columns]
            
            if missing_columns:
                return False, f"Missing required columns in sheet '{sheet}': {', '.join(missing_columns)}!"
        
        return True, ""  # File is valid
    except Exception as e:
        return False, f"Error reading the file: {str(e)}"
    
# Set page config
st.set_page_config(page_title="AB Supply Chain Optimization", layout="wide")

# Title
st.title("AB Supply Chain Optimization and Visualization")

tab1, tab2, tab3 = st.tabs(["1. Run Optimization", "2. Visualize Results", "3. Visualize Suggested Routes"])

with tab1:
    st.header("Run Production and Shipping Optimization")

    # Display the supply chain map
    try:
        supply_chain_map = create_supply_chain_map(df_plants, df_warehouses)
        st.components.v1.html(supply_chain_map._repr_html_(), height=800)
    except Exception as e:
        st.error(f"Error generating supply chain map: {str(e)}")
    
    # Use session state to maintain data across reruns
    if 'optimization_results' not in st.session_state:
        st.session_state['optimization_results'] = None
        st.session_state['monthly_data'] = None
        st.session_state['results'] = None

    uploaded_file = st.file_uploader("Upload `Solver_Sales_Monthly.xlsx` that contains monthly sales (demand), shipping costs, production costs, and production capacity data:", type=['xlsx'])
    
    if uploaded_file is not None:
        # Validate the file
        is_valid, error_message = validate_uploaded_file(uploaded_file)
        if not is_valid:
            st.error(f"File validation failed: {error_message}")
        else:
            if st.button("Run Optimization"):
                with st.spinner("Running optimization for all months..."):
                    try:
                        # Run optimization
                        results = optimize_all_months(uploaded_file, mappings)
                        
                        if results is not None:
                            # Create BytesIO object for Excel file
                            output = io.BytesIO()
                            
                            # Save results to Excel file in memory
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                # Save monthly data sheets
                                for monthly_data in results['monthly_data']:
                                    monthly_data['data'].to_excel(writer, 
                                        sheet_name=f"Data-{monthly_data['month']}", 
                                        index=False)
                                
                                # Save summary sheets
                                results['results_df'].to_excel(writer, sheet_name='Summary', index=False)
                                results['percentage_summary'].to_excel(writer, sheet_name='Percentage Summary', index=False)
                                results['annual_summary'].to_excel(writer, sheet_name='Annual Summary', index=False)
                            
                            output.seek(0)  # Reset file pointer to beginning
                            
                            # Store results in session state
                            st.session_state['optimization_results'] = output
                            st.session_state['monthly_data'] = results['monthly_data']
                            st.session_state['results'] = results
                            
                            # Display optimization summary
                            st.success("Optimization complete!")
                        else:
                            st.error("Optimization failed. Please check your input data.")
                    except Exception as e:
                        st.error(f"An error occurred during optimization: {str(e)}")
    
    # Show download button if optimization results exist
    if st.session_state['optimization_results'] is not None:
        st.download_button(
            label="Download Results",
            data=st.session_state['optimization_results'].getvalue(),
            file_name="Optimization_Results_Monthly.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
with tab2:
    st.header("Visualize Optimization Results")
    
    if 'optimization_results' not in st.session_state or st.session_state['optimization_results'] is None:
        st.warning("Please run optimization first in the **Run Optimization** tab.")
    else:
        try:
            # Create placeholder for plots
            plot_container = st.container()
            
            with plot_container:                
                # Reset file pointer and read Summary sheet
                st.session_state['optimization_results'].seek(0)
                results_df = st.session_state['results']['results_df']
                
                # Generate plots
                create_optimization_plots(results_df, "./")
                
                # Display plots in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    try:
                        st.image("total_cost_current.png")
                        st.image("total_cost_all.png")
                        st.image("production_cost_comparison.png")
                        st.image("savings_comparison_by_month.png")
                    except Exception as e:
                        st.error(f"Error loading cost comparison plots: {str(e)}")
                
                with col2:
                    try:
                        st.image("total_cost_current_base.png")
                        st.image("shipping_cost_comparison.png")
                        st.image("savings_comparison_by_product.png")
                    except Exception as e:
                        st.error(f"Error loading savings comparison plots: {str(e)}")
            
            # Display overall statistics
            st.header("Optimization Statistics")
            # Extract annual summary
            annual_summary = st.session_state['results']['annual_summary']
            
            # Extract costs and savings
            current_cost = annual_summary['current_cost'].iloc[0]
            current_production_cost = annual_summary['current_production'].iloc[0]
            current_shipping_cost = annual_summary['current_shipping'].iloc[0]
            
            base_total_cost = annual_summary['base_total_cost'].iloc[0]
            base_production_cost = annual_summary['base_production_cost'].iloc[0]
            base_shipping_cost = annual_summary['base_shipping_cost'].iloc[0]
            
            rules_total_cost = annual_summary['base_and_rules_total_cost'].iloc[0]
            rules_production_cost = annual_summary['base_and_rules_production_cost'].iloc[0]
            rules_shipping_cost = annual_summary['base_and_rules_shipping_cost'].iloc[0]
            
            # Create a DataFrame for cost metrics
            cost_comparison_df = pd.DataFrame({
                "Cost Type": ["Total Cost", "Production Cost", "Shipping Cost"],
                "Current": [
                    format_number(current_cost),
                    format_number(current_production_cost),
                    format_number(current_shipping_cost)
                ],
                "Base Optimization": [
                    format_number(base_total_cost),
                    format_number(base_production_cost),
                    format_number(base_shipping_cost)
                ],
                "Base + Rules Optimization": [
                    format_number(rules_total_cost),
                    format_number(rules_production_cost),
                    format_number(rules_shipping_cost)
                ]
            })
            
            # Display the DataFrame
            st.subheader("Cost Comparison")
            st.dataframe(cost_comparison_df, use_container_width=True)

            # Calculate savings metrics
            base_savings = annual_summary['base_total_savings'].iloc[0]
            base_savings_percent = (base_savings / current_cost) * 100
            rules_savings = annual_summary['base_and_rules_total_savings'].iloc[0]
            rules_savings_percent = (rules_savings / current_cost) * 100

            # Create a DataFrame for savings summary
            savings_summary_df = pd.DataFrame({
                "Optimization Type": ["Base Optimization", "Base + Rules Optimization"],
                "Average Monthly Savings": [
                    format_number(base_savings / 12),
                    format_number(rules_savings / 12)
                ],
                "Average Savings Percentage": [
                    f"{base_savings_percent:.1f}%",
                    f"{rules_savings_percent:.1f}%"
                ],
                "Total Annual Savings": [
                    format_number(base_savings),
                    format_number(rules_savings)
                ]
            })

            # Display the DataFrame
            st.subheader("Savings Summary")
            st.dataframe(savings_summary_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating visualization: {str(e)}")

with tab3:
    st.header("Visualize Suggested Shipping Routes")
    
    if 'optimization_results' not in st.session_state or st.session_state['optimization_results'] is None:
        st.warning("Please run optimization first in the **Run Optimization** tab.")
    else:
        try:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_product = st.selectbox(
                    "Select Product",
                    options=list(mappings['products'].values()),
                    index=3 # Default to Lids
                )
                product_id = {v: k for k, v in mappings['products'].items()}[selected_product]
            
            with col2:
                selected_month = st.selectbox(
                    "Select Month",
                    options=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                )
            
            with col3:
                base_only = st.radio(
                    "Optimization Type",
                    options=["Base Only", "Base + Rules"],
                    index=1 # Default to Base Only
                ) == "Base Only"
            
            if st.button("Generate Visualization"):
                try:
                    with st.spinner("Generating shipping routes visualization..."):
                        monthly_data = st.session_state['monthly_data']
                        
                        st.write(f"Suggested Routes for {selected_product} in {selected_month} under {'Base Only' if base_only else 'Base + Rules'} optimization:")
                        
                        # Verify data
                        month_data = next((data['data'] for data in monthly_data if data['month'] == selected_month), None)
                        if month_data is None:
                            st.error(f"No data found for month: {selected_month}")
                            st.stop()
                            
                        # Filter for specific product and verify data exists
                        df_product = month_data[month_data['i'] == product_id].copy()
                        if df_product.empty:
                            st.error(f"No data found for product {selected_product} in {selected_month}")
                            st.stop()
                        
                        # Generate map and routes
                        shipping_routes_map, routes_df = create_shipping_routes_map(
                            df_plants, 
                            df_warehouses, 
                            monthly_data,
                            product_id=product_id, 
                            base_only=base_only, 
                            month=selected_month, 
                            mappings=mappings
                        )
                        
                        # Display route changes if available
                        if not routes_df.empty:
                            # Display the map
                            st.components.v1.html(shipping_routes_map._repr_html_(), height=600)

                            st.header("Suggested Shipping Routes Details")
                            diff_col = 'diff_base' if base_only else 'diff_base_and_rules'
                            display_cols = ['plant_code', 'warehouse_id', 'warehouse_name', 'associated_customers', diff_col]
                            routes_display = routes_df[display_cols].copy()
                            routes_display[diff_col] = routes_display[diff_col].apply(lambda x: f"+{float(x):.3f}" if float(x) > 0 else f"{float(x):.3f}")
                            routes_display.columns = ['Plant Code', 'Warehouse #', 'Warehouse Name', 'Warehouse Customers', 'Shipment Adjustment (Million Units)'] # rename columns
                            
                            # Create a styled dataframe
                            def style_diff(val):
                                if pd.isna(val):
                                    return ''
                                color = 'red' if float(val) < 0 else 'green'
                                return f'color: {color}'
                            
                            styled_df = routes_display.reset_index(drop=True).style.map(style_diff, subset=['Shipment Adjustment (Million Units)'])
                            st.dataframe(styled_df, use_container_width=True)
                        else:
                            st.info("No route changes found for the selected criteria.")
                            
                except Exception as e:
                    st.error(f"Error generating shipping routes visualization: {str(e)}")
                    st.write("Error details:", str(e))
                    import traceback
                    st.code(traceback.format_exc())

        except Exception as e:
            st.error(f"Error setting up shipping routes tab: {str(e)}")

# Add instructions
with st.sidebar:
    with st.expander("Instructions"): # Use auto-width by default
        st.write("""
        ### Step 1: Run Optimization
        1. Upload the input data file `Solver_Sales_Monthly.xlsx` containing monthly sales (demand), shipping costs, production costs, and production capacity data
        2. Click "Run Optimization" to process data and generate optimal shipping plans for all months
        3. Download the complete results in `Optimization_Results_Monthly.xlsx`
        
        ### Step 2: View Optimization Analysis
        1. After running optimization, navigate to the "Visualize Results" tab
        2. Review the generated cost comparison charts and savings analysis
        3. Examine the detailed breakdowns of:
           - Total costs (current vs optimized)  
           - Production and shipping cost changes
           - Monthly and product-specific savings
        
        ### Step 3: Explore Shipping Routes
        1. Navigate to the "Visualize Suggested Routes" tab
        2. Select specific parameters:
           - Product type (e.g., Lids, 12oz cans)
           - Month to analyze 
           - Optimization type (Base Only vs Base + Rules)
        3. Click "Generate Visualization" to view:
           - Interactive map showing suggested route changes
           - Green arrows: increased shipment volume
           - Red arrows: decreased shipment volume
           - Detailed breakdown of warehouse allocations
        """)