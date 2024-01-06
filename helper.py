#Given code from labs

import matplotlib.pyplot as plt
import numpy as np
import csv
import random
import os
import math

from BSE import *

# Use this to plot trades of a single experiment
def plot_trades(trial_id):
    prices_fname = trial_id + '_tape.csv'
    x = np.empty(0)
    y = np.empty(0)
    with open(prices_fname, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            time = float(row[1])
            price = float(row[2])
            x = np.append(x,time)
            y = np.append(y,price)

    plt.plot(x, y, 'x', color='black') 
    
# Use this to run an experiment n times and plot all trades
def n_runs_plot_trades(n, trial_id, start_time, end_time, traders_spec, order_sched):
    x = np.empty(0)
    y = np.empty(0)

    for i in range(n):
        trialId = trial_id + '_' + str(i)
        tdump = open(trialId + '_avg_balance.csv','w')

        market_session(trialId, start_time, end_time, traders_spec, order_sched, tdump, True, False)
        
        tdump.close()

        with open(trialId + '_tape.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                time = float(row[1])
                price = float(row[2])
                x = np.append(x,time)
                y = np.append(y,price)

    plt.plot(x, y, 'x', color='black');

# !!! Don't use on it's own   
def getorderprice(i, sched, n, mode):
    pmin = min(sched[0][0], sched[0][1])
    pmax = max(sched[0][0], sched[0][1])
    prange = pmax - pmin
    stepsize = prange / (n - 1)
    halfstep = round(stepsize / 2.0)

    if mode == 'fixed':
        orderprice = pmin + int(i * stepsize)
    elif mode == 'jittered':
        orderprice = pmin + int(i * stepsize) + random.randint(-halfstep, halfstep)
    elif mode == 'random':
        if len(sched) > 1:
            # more than one schedule: choose one equiprobably
            s = random.randint(0, len(sched) - 1)
            pmin = min(sched[s][0], sched[s][1])
            pmax = max(sched[s][0], sched[s][1])
        orderprice = random.randint(pmin, pmax)
    return orderprice    

# !!! Don't use on it's own
def make_supply_demand_plot(bids, asks):
    # total volume up to current order
    volS = 0
    volB = 0

    fig, ax = plt.subplots()
    plt.ylabel('Price')
    plt.xlabel('Quantity')
    
    pr = 0
    for b in bids:
        if pr != 0:
            # vertical line
            ax.plot([volB,volB], [pr,b], 'r-')
        # horizontal lines
        line, = ax.plot([volB,volB+1], [b,b], 'r-')
        volB += 1
        pr = b
    if bids:
        line.set_label('Demand')
        
    pr = 0
    for s in asks:
        if pr != 0:
            # vertical line
            ax.plot([volS,volS], [pr,s], 'b-')
        # horizontal lines
        line, = ax.plot([volS,volS+1], [s,s], 'b-')
        volS += 1
        pr = s
    if asks:
        line.set_label('Supply')
        
    if bids or asks:
        plt.legend()
    plt.show()

# Use this to plot supply and demand curves from supply and demand ranges and stepmode
def plot_sup_dem(seller_num, sup_ranges, buyer_num, dem_ranges, stepmode):
    asks = []
    for s in range(seller_num):
        asks.append(getorderprice(s, sup_ranges, seller_num, stepmode))
    asks.sort()
    bids = []
    for b in range(buyer_num):
        bids.append(getorderprice(b, dem_ranges, buyer_num, stepmode))
    bids.sort()
    bids.reverse()
    
    make_supply_demand_plot(bids, asks) 

# plot sorted trades, useful is some situations - won't be used in this worksheet
def in_order_plot(trial_id):
    prices_fname = trial_id + '_tape.csv'
    y = np.empty(0)
    with open(prices_fname, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            price = float(row[2])
            y = np.append(y,price)
    y = np.sort(y)
    x = list(range(len(y)))

    plt.plot(x, y, 'x', color='black')   

# plot offset function
def plot_offset_fn(offset_fn, total_time_seconds):   
    x = list(range(total_time_seconds))
    offsets = []
    for i in range(total_time_seconds):
        offsets.append(offset_fn(i))
    plt.plot(x, offsets, 'x', color='black')  


#Part A code chunks 

import glob
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

import glob
import pandas as pd

def read_csv(file_path):
    profit_by_shvr = []
    profit_by_zic = []
    overall_winners = {'SHVR': 0, 'ZIC': 0, 'Tie': 0}

    csv_files = glob.glob(file_path)
    for file in csv_files:
        df = pd.read_csv(file, usecols=range(12))
        trader_type_col, profit_col = 4, 7

        shvr_profit = 0
        zic_profit = 0

        while trader_type_col < len(df.columns) and profit_col < len(df.columns):
            trader_type = str(df.iloc[-1, trader_type_col]).strip()
            profit = pd.to_numeric(df.iloc[-1, profit_col], errors='coerce')

            if pd.isna(profit):
                trader_type_col += 4
                profit_col += 4
                continue

            if trader_type == 'SHVR':
                shvr_profit += profit
                profit_by_shvr.append(profit)
            elif trader_type == 'ZIC':
                zic_profit += profit
                profit_by_zic.append(profit)

            trader_type_col += 4
            profit_col += 4

        # Determine the winner for this trial (file)
        if shvr_profit > zic_profit:
            overall_winners['SHVR'] += 1
        elif shvr_profit < zic_profit:
            overall_winners['ZIC'] += 1
        else:
            overall_winners['Tie'] += 1

    return profit_by_shvr, profit_by_zic, overall_winners


def perform_shapirowilktest(data):
    _, p_val = stats.shapiro(data)
    
    if p_val > 0.05:
        null_hypothesis_finding = "Failed to reject H₀. Data appears to be normally distributed."
        alternative_hypothesis_finding = "Not enough evidence to support the data is not normally distributed."
    else:
        null_hypothesis_finding = "Reject H₀. Data does not appear to be normally distributed."
        alternative_hypothesis_finding = "Evidence supports the data is not normally distributed."

    return p_val, null_hypothesis_finding, alternative_hypothesis_finding


def normalize(df, column_names):
    result = df.copy()
    for feature_name in column_names:
        mean_value = df[feature_name].mean()
        std_value = df[feature_name].std()
        result[feature_name] = (df[feature_name] - mean_value) / std_value
    return result

def perform_kstest(data):
    _, p_val = stats.kstest(data, 'norm')
    print(f"Kolmogorov-Smirnov test p-value: {p_val}")
    return p_val

def perform_ttest(data1, data2, alternative: str = 'greater'):
    t_stat, p_val = stats.ttest_rel(data1, data2, alternative=alternative)

    if p_val > 0.05:
        null_hypothesis_finding = "Failed to reject H₀. No significant difference in means."
        alternative_hypothesis_finding = "Not enough evidence to support a significant difference in means."
    else:
        null_hypothesis_finding = "Reject H₀. Significant difference in means."
        alternative_hypothesis_finding = "Evidence supports the alternative hypothesis."

    return t_stat, p_val, null_hypothesis_finding, alternative_hypothesis_finding


def perform_wilcoxon(data1, data2, alternative: str = 'greater'):
    t_stat, p_val = stats.wilcoxon(data1, data2, alternative=alternative)
    
    if p_val > 0.05:
        null_hypothesis_finding = "Failed to reject H₀. No significant difference."
        alternative_hypothesis_finding = "Not enough evidence to support the alternative hypothesis."
    else:
        null_hypothesis_finding = "Reject H₀. Significant difference."
        alternative_hypothesis_finding = "Evidence supports the alternative hypothesis."

    return t_stat, p_val, null_hypothesis_finding, alternative_hypothesis_finding


def plot_kde(profit_by_shvr, profit_by_zic, x_value, ax):
    sns.kdeplot(profit_by_shvr, ax=ax, color='blue', label='SHVR')
    sns.kdeplot(profit_by_zic, ax=ax, color='orange', label='ZIC')
    ax.set_title(f'x={x_value}')

def sample_data(data, sample_size=10):
    """Randomly sample data if it's larger than the specified sample size."""
    if len(data) > sample_size:
        return np.random.choice(data, size=sample_size, replace=False)
    return data

def create_qq_plots(data_shvr, data_zic, n, axs):
    """Create Q-Q plots for SHVR and ZIC data."""
    sampled_shvr = sample_data(data_shvr)
    sampled_zic = sample_data(data_zic)

    plot_qq(sampled_shvr, axs[0], f'SHVR_{n}')
    plot_qq(sampled_zic, axs[1], f'ZIC_{n}')

def plot_qq(data, ax, label):
    """Generate a Q-Q plot."""
    stats.probplot(data, dist="norm", plot=ax)
    ax.set_title(f'Q-Q Plot for {label}')


#Part B code chunks

import glob
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import glob
import pandas as pd



def read_csv_multi(x_values, file_pattern):
    profit_by_shvr = {x: [] for x in x_values}
    profit_by_zic = {x: [] for x in x_values}
    overall_winners = {x: {'SHVR': 0, 'ZIC': 0, 'Tie': 0} for x in x_values}

    for x in x_values:
        csv_files = glob.glob(file_pattern.format(x=x))


        for file in csv_files:
            df = pd.read_csv(file, usecols=range(12))
            shvr_file_profit, zic_file_profit = process_file(df)
            profit_by_shvr[x].append(shvr_file_profit)
            profit_by_zic[x].append(zic_file_profit)
            
            #winner at this file
            if shvr_file_profit > zic_file_profit:
                overall_winners[x]['SHVR'] += 1
            elif shvr_file_profit < zic_file_profit:
                overall_winners[x]['ZIC'] += 1
            else:
                overall_winners[x]['Tie'] += 1

        print(f"Results for x = {x}: SHVR wins: {overall_winners[x]['SHVR']}, ZIC wins: {overall_winners[x]['ZIC']}, Ties: {overall_winners[x]['Tie']}")

    return profit_by_shvr, profit_by_zic, overall_winners

def process_file(df):
    shvr_profit = 0
    zic_profit = 0
    trader_type_col, profit_col = 4, 7

    while trader_type_col < len(df.columns) and profit_col < len(df.columns):
        trader_type = str(df.iloc[-1, trader_type_col]).strip()
        profit = pd.to_numeric(df.iloc[-1, profit_col], errors='coerce')

        if pd.isna(profit):
            trader_type_col += 4
            profit_col += 4
            continue

        if trader_type == 'SHVR':
            shvr_profit += profit
        elif trader_type == 'ZIC':
            zic_profit += profit

        trader_type_col += 4
        profit_col += 4

    return shvr_profit, zic_profit


def perform_normality_tests(x_values, profit_by_shvr, profit_by_zic):
    ncols = 5
    nrows = len(x_values) // ncols + (len(x_values) % ncols > 0)
    fig, axs = plt.subplots(nrows, ncols, figsize=(10, 1.5*nrows), sharex=True, sharey=True)
    axs = axs.flatten()

    results = []

    for i, x in enumerate(x_values):
        if profit_by_shvr[x] and profit_by_zic[x]:
            shvr_p = perform_shapirowilktest(profit_by_shvr[x])
            zic_p = perform_shapirowilktest(profit_by_zic[x])
            plot_kde(profit_by_shvr[x], profit_by_zic[x], x, axs[i])

            results.append({
                'X Value': x,
                'SHVR p-value': shvr_p,
                'ZIC p-value': zic_p,
            })
        else:
            axs[i].axis('off')

    blue_patch = Patch(color='blue', label='SHVR')
    orange_patch = Patch(color='orange', label='ZIC')
    plt.legend(handles=[blue_patch, orange_patch], loc='upper right')
    plt.tight_layout()
    plt.show()

    return pd.DataFrame(results)

#Part C code chunks

def get_trader_spec(ratios, total_num):
    return [(trader, int(ratio * total_num)) for trader, ratio in ratios.items()]


def run_section(section_id, permutations, total_num):
    section_folder = f'part_c_{section_id}'
    os.makedirs(section_folder, exist_ok=True)

    for perm in permutations:
        perm_folder_name = "_".join(f"{int(ratio * 100)}" for ratio in perm)
        perm_output_folder = os.path.join(section_folder, perm_folder_name)
        os.makedirs(perm_output_folder, exist_ok=True)

        ratios = dict(zip(['ZIC', 'SHVR', 'ZIP', 'GVWY'], perm))
        sellers_spec = get_trader_spec(ratios, total_num)
        buyers_spec = sellers_spec
        traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

        for i in range(10):
            trial_id = os.path.join(perm_output_folder, f"trial_{i}")
            output = market_session(trial_id, start_time, end_time, traders_spec, order_sched, dump_flags, verbose)

import glob
import pandas as pd
from collections import defaultdict
from scipy import stats


def process_profit_data(df, profitabilities):
    trader_type_col = 4 
    profit_col = 7       

    row_index = -1 if len(df.index) > 1 else 0

    while trader_type_col < len(df.columns) and profit_col < len(df.columns):
        trader_type = str(df.iloc[row_index, trader_type_col]).strip()
        profit = pd.to_numeric(df.iloc[row_index, profit_col], errors='coerce')

        if pd.notna(profit):
            profit = float(profit)
            profitabilities[trader_type].append(profit)

        trader_type_col += 4
        profit_col += 4

def read_and_process_files(sections, num_trials, base_path):
    section_profitabilities = {section_id: defaultdict(lambda: defaultdict(list)) for section_id in sections}
    for section_id, permutations in sections.items():
        for perm in permutations:
            perm_folder_name = "_".join(str(int(ratio)) for ratio in perm)
            for trial in range(num_trials):
                file_pattern = f"{base_path}_{section_id}/{perm_folder_name}/trial_{trial}_avg_balance.csv"
                csv_files = glob.glob(file_pattern)
                for file_path in csv_files:
                    df = pd.read_csv(file_path,header=None, usecols=range(20), low_memory=False)
                    process_profit_data(df, section_profitabilities[section_id][perm_folder_name])
    
    return section_profitabilities


def calculate_average_profitability(profitabilities):
    return {trader: sum(profits) / len(profits) for trader, profits in profitabilities.items() if profits}

def perform_pairwise_t_tests(trader_profits):
    for trader1, profits1 in trader_profits.items():
        for trader2, profits2 in trader_profits.items():
            if trader1 != trader2:
                t_stat, p_val = stats.ttest_ind(profits1, profits2, equal_var=False)
                print(f"t-test between {trader1} and {trader2}: t = {t_stat}, p = {p_val}")

def perform_friedman_test(profitabilities_by_section):
    data_for_test = []
    for section in profitabilities_by_section.values():
        trader_profits = list(section.values())
        min_length = min(map(len, trader_profits))
        data_for_test.append([profits[:min_length] for profits in trader_profits])
    
    transposed_data = list(map(list, zip(*data_for_test)))

    stat, p = stats.friedmanchisquare(*transposed_data)
    return stat, p

def perform_friedman_test_single_section(trader_profits):
    min_length = min(len(profits) for profits in trader_profits.values())
    data_for_test = [profits[:min_length] for profits in trader_profits.values()]

    stat, p = stats.friedmanchisquare(*data_for_test)
    return stat, p

import pandas as pd
from scipy import stats

def perform_friedman_test_single_permutation(trader_profits):
    min_length = min(len(profits) for profits in trader_profits.values())
    aligned_data = [profits[:min_length] for profits in trader_profits.values()]

    stat, p = stats.friedmanchisquare(*aligned_data)

    if p > 0.05:
        null_hypothesis_finding = "Failed to reject H₀. No significant difference."
        alternative_hypothesis_finding = "Not enough evidence to support the alternative hypothesis."
    else:
        null_hypothesis_finding = "Reject H₀. Significant difference."
        alternative_hypothesis_finding = "Evidence supports the alternative hypothesis."

    return p, null_hypothesis_finding, alternative_hypothesis_finding


def perform_friedman_tests_by_section(section_profitabilities):
    results = []

    for section_id, permutations in section_profitabilities.items():
        for perm_name, trader_profits in permutations.items():
            p, reject, find= perform_friedman_test_single_permutation(trader_profits)
            results.append({
                'Section': section_id,
                'Permutation': perm_name,
                'Friedman Statistic': reject,
                'Meaning': find,
                'P-Value': p
            })

    return pd.DataFrame(results)

def plot_kde_for_permutation(ax, trader_profits, section_id, perm_name):

    for trader_type, profits in trader_profits.items():
        sns.kdeplot(profits, ax=ax, label=trader_type)

    ax.set_title(f"Section {section_id} - {perm_name}")
    ax.set_xlabel('Profit')
    ax.set_ylabel('Density')
    ax.legend()

def plot_all_kdes(section_profitabilities, title):
    num_sections = len(section_profitabilities)
    num_permutations = max(len(perms) for perms in section_profitabilities.values())
    total_plots = num_sections * num_permutations
    ncols = 4 

    
    nrows = total_plots // ncols + (total_plots % ncols > 0)
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*4))
    axs = axs.flatten()

    plot_index = 0
    for section_id, permutations in section_profitabilities.items():

        for perm_name, trader_profits in permutations.items():
            ax = axs[plot_index]
            plot_kde_for_permutation(ax, trader_profits, section_id, perm_name)
            plot_index += 1

    for i in range(plot_index, len(axs)):
        axs[i].axis('off')

    plt.suptitle(title, fontsize=20, y=1.02) 
    plt.tight_layout()
    plt.show()

def is_normal_distribution(trader_profits):
    for profits in trader_profits.values():
        _, p_val = stats.shapiro(profits)
        if p_val <= 0.05:  
            return False
    return True

def perform_anova_test(trader_profits):
    aligned_data = [profits for profits in trader_profits.values()]
    f_stat, p_val = stats.f_oneway(*aligned_data)
    return f_stat, p_val

def perform_tests_by_section(section_profitabilities):
    results = []

    for section_id, permutations in section_profitabilities.items():
        for perm_name, trader_profits in permutations.items():
            if is_normal_distribution(trader_profits):
                p_val = perform_anova_test(trader_profits)
                test_type = "ANOVA"
                normal = "Normal"
                null_hypothesis_finding = "Assumption not specified for ANOVA"
                alternative_hypothesis_finding = "Assumption not specified for ANOVA"
            else:
                p_val, null_hypothesis_finding, alternative_hypothesis_finding = perform_friedman_test_single_permutation(trader_profits)
                normal = "Not Normal"
                test_type = "Friedman"

            results.append({
                'Section': section_id,
                'Permutation': perm_name,
                'Test Type': test_type,
                'Normality': normal,
                'P-Value': p_val,
                'Null Hypothesis Finding': null_hypothesis_finding,
                'Alternative Hypothesis Finding': alternative_hypothesis_finding
            })

    return pd.DataFrame(results)

def perform_1vALL(section_profitabilities, title, alpha=0.05):
    results = []

    for section_id, permutations in section_profitabilities.items():
        for perm_name, trader_profits in permutations.items():
            combined_profits = np.concatenate(list(trader_profits.values()))
            for trader, profits in trader_profits.items():
                other_profits = combined_profits[~np.isin(combined_profits, profits)]
                min_length = min(len(profits) for profits in trader_profits.values())
                aligned_data = [profits[:min_length] for profits in trader_profits.values()]

                if is_normal_distribution({trader: profits, 'ALL_OTHERS': other_profits}):
                    f_stat, p_val = stats.f_oneway(profits, other_profits)
                    test_type = "ANOVA"
                else:
                    stat, p_val = stats.friedmanchisquare(*aligned_data)
                    test_type = "Friedman"
                
                results.append({
                    'Section': section_id,
                    'Permutation': perm_name,
                    'Trader Type': trader,
                    'Test Type': test_type,
                    'P-Value': p_val,
                    'Is Significant Before': p_val <= alpha
                })

    
from statsmodels.stats.multitest import multipletests


def perform_1vALL(section_profitabilities, title, alpha=0.05):
    results = []

    for section_id, permutations in section_profitabilities.items():
        for perm_name, trader_profits in permutations.items():
            combined_profits = np.concatenate(list(trader_profits.values()))
            for trader, profits in trader_profits.items():
                other_profits = combined_profits[~np.isin(combined_profits, profits)]
                min_length = min(len(profits) for profits in trader_profits.values())
                aligned_data = [profits[:min_length] for profits in trader_profits.values()]

                if is_normal_distribution({trader: profits, 'ALL_OTHERS': other_profits}):
                    f_stat, p_val = stats.f_oneway(profits, other_profits)
                    test_type = "ANOVA"
                else:
                    stat, p_val = stats.friedmanchisquare(*aligned_data)
                    test_type = "Friedman"
                
                results.append({
                    'Section': section_id,
                    'Permutation': perm_name,
                    'Trader Type': trader,
                    'Test Type': test_type,
                    'P-Value': p_val,
                    'Is Significant Before': p_val <= alpha
                })
                
    results_df = pd.DataFrame(results)

    p_values = results_df['P-Value']
    p_reject, p_adjusted,  _, _ = multipletests(p_values, alpha=alpha, method='bonferroni')

    results_df['Adjusted P-Value'] = p_adjusted
    results_df['Is Significant After'] = p_reject  

    print(title)

    return results_df


def plot_section_profitability_heatmap(section_profitabilities, title):
    data_list = []

    for section, permutations in section_profitabilities.items():
        for permutation, trader_profits in permutations.items():
            for trader, profits in trader_profits.items():
                avg_profit = np.mean(profits)  
                data_list.append({
                    'Section': section,
                    'Permutation': permutation,
                    'Trader Type': trader,
                    'Average Profit': avg_profit
                })

    df = pd.DataFrame(data_list)

    pivot_df = df.pivot_table(index=['Section', 'Permutation'], columns='Trader Type', values='Average Profit')

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_df, annot=True, cmap='coolwarm', fmt=".1f")
    x=(f'Profit of Traders Across Permutations for {title}')
    plt.title(x)
    plt.ylabel('Section and Permutation')
    plt.xlabel('Trader Types')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return df


#Part D code chunks
import pandas as pd
import matplotlib.pyplot as plt

def adjust_for_resets(data):
 
    adjusted_data = []
    last_non_reset_value = 0
    total_offset = 0

    for value in data:
        if value < last_non_reset_value: 
            total_offset += last_non_reset_value
        adjusted_value = value + total_offset
        adjusted_data.append(adjusted_value)
        last_non_reset_value = value

    return adjusted_data

def create_dataframe(data, column_prefix):
   
    df = pd.DataFrame(data).transpose()
    df.columns = [f'Trial {i+1} {column_prefix}' for i in range(len(df.columns))]

    for column in df.columns:
        df[column] = adjust_for_resets(df[column])

    return df

def plot_data(df_zipsh, df_zic):
 
    plt.figure(figsize=(15, 6))

    # Plot for ZIPSH
    for column in df_zipsh.columns:
        plt.plot(df_zipsh[column], marker='', label=column )

    # Plot for ZIC
    for column in df_zic.columns:
        plt.plot(df_zic[column], marker='', label=column,linestyle=':')

    plt.title('Profit Comparison by Trader Type and Trial')
    plt.xlabel('Data Points')
    plt.ylabel('Profit')
    plt.legend()
    plt.show()


profit_by_zipsh = [[] for _ in range(5)]
profit_by_zic = [[] for _ in range(5)]

for i in range(5):
 
    file_path = f"part_d1_50_30D_1/trial_{i}_avg_balance.csv"
    csv_files = glob.glob(file_path)
    
    for file in csv_files:
        df = pd.read_csv(file, usecols=range(12))
        
     
        for index, row in df.iterrows():

            trader_type_col = 4
            profit_col = 7


            while trader_type_col < len(df.columns) and profit_col < len(df.columns):
              
                trader_type = str(row.iloc[trader_type_col]).strip()
                profit = pd.to_numeric(row.iloc[profit_col], errors='coerce')       
                
                if pd.isna(profit):
                    trader_type_col += 4
                    profit_col += 4
                    continue

                if trader_type == 'ZIPSH':
                    profit_by_zipsh[i].append(profit)
                elif trader_type == 'ZIC':
                    profit_by_zic[i].append(profit)

    
                trader_type_col += 4
                profit_col += 4

import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import re

def read_data_prof(trials, file_pattern):
    dataframes = []
    for trial in trials:
        for file in glob.glob(file_pattern.format(trial=trial)):
            times, best_B_profs = process_line_prof(file)
            if times and best_B_profs:
                dataframes.append(pd.DataFrame({
                    'Trial': [f'Trial {trial}'] * len(times), 
                    'Time (days)': times, 
                    'Profit per Second': best_B_profs
                }))
    return pd.concat(dataframes)

def process_line_prof(file_path):
    best_B_prof_values, t_values = [], []
    with open(file_path, 'r') as file:
        for line in file:
            t_match = re.search(r't=,(\d+)', line)
            best_B_prof_match = re.search(r'best_B_prof=,([\d.-]+)', line)
            if best_B_prof_match and t_match:
                best_B_prof_values.append(float(best_B_prof_match.group(1)))
                t_values.append(int(t_match.group(1)))
    return t_values, best_B_prof_values

def read_data_param(trials, file_pattern):
    dataframes = []
    for trial in trials:
        times, betas, momentums, mBuys, c_as, c_rs  = [], [], [], [], [],[]
        csv_files = glob.glob(file_pattern.format(trial=trial))
        for file in csv_files:
            with open(file, 'r') as f:
                for line in f:
                    pairs = line.split(", ")
                    time, beta, momentum, mBuy, c_a, c_r = process_line_param(pairs, file) 
                    if beta is not None and time is not None and momentum is not None and mBuy is not None and c_a is not None and c_r is not None:
                        times.append(time)
                        betas.append(beta)
                        momentums.append(momentum)
                        mBuys.append(mBuy)
                        c_as.append(c_a)
                        c_rs.append(c_r)
        dataframes.append(pd.DataFrame({'Trial': [f'Trial {trial}'] * len(betas),'Time (days)': times,  'Beta': betas, 'Momentum': momentums, 'mBuy': mBuys, 'c_a': c_as, 'c_r': c_rs}))
    return pd.concat(dataframes)




def process_line_param(pairs, file):
    time, beta, momentum, mBuy, c_a, c_r = None, None, None, None, None, None
    for pair in pairs:
        key, _, value = pair.partition('=')
        value = value.strip().replace(',', '')
        if key.strip() == 't':
            time = parse_value(value, file)
        elif key.strip() == 'b':
            beta = parse_value(value, file)
        elif key.strip() == 'ca':
            c_a = parse_value(value, file)
        elif key.strip() == 'cr':
            c_r = parse_value(value, file)
        elif key.strip() == 'm':
            momentum = parse_value(value, file)
        elif key.strip() == 'mBuy':
            mBuy = parse_value(value, file)
    return time, beta, momentum, mBuy, c_a, c_r


def parse_value(value, file):
    try:
        return float(value) / (60 * 60 * 24)
    except ValueError:
        print(f"Invalid value: {value} in file: {file}")
        return None

def normalize_time(data, number_of_days):
    data['Time (days)'] = pd.to_numeric(data['Time (days)'], errors='coerce')
    max_t_value = data['Time (days)'].max()

    #print("max_t_value:", max_t_value)

    if pd.notnull(max_t_value) and max_t_value != 0:

        data['Normalized Time'] = (data['Time (days)'] / max_t_value) * number_of_days
        data['Discrete Time (days)'] = np.round(data['Normalized Time'])
    else:
        print("Error: Invalid or missing time data.")


def plot_data(data, colors):
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    
    
    # If more than 10 trials, sample 10 equally spaced trials
    unique_trials = data['Trial'].unique()
    
    if len(unique_trials) > 10:

        sample_indices = np.linspace(0, len(unique_trials) - 1, 10, dtype=int)

        trials_to_plot = unique_trials[sample_indices]

        data = data[data['Trial'].isin(trials_to_plot)]

    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, len(data['Trial'].unique())))

    # Raw data
    sns.lineplot(ax=axs[0], data=data, x='Time (days)', y='Profit per Second', hue='Trial')
    axs[0].set_xlabel('Time (days)')
    axs[0].set_ylabel('Profit per Second')
    axs[0].set_title('Raw Data Plot')

    # Normalized data
    sns.lineplot(ax=axs[1], data=data, x='Discrete Time (days)', y='Profit per Second', hue='Trial')
    axs[1].set_xlabel('Discrete Time (days)')
    axs[1].set_ylabel('Profit per Second')
    axs[1].set_title('Normalized Data Plot')

    # Average with variance
    grouped_data = data.groupby('Discrete Time (days)')['Profit per Second'].agg(['mean', 'std'])
    grouped_data['upper'] = grouped_data['mean'] + grouped_data['std']
    grouped_data['lower'] = grouped_data['mean'] - grouped_data['std']
    axs[2].plot(grouped_data.index, grouped_data['mean'], label='Average PPS')
    axs[2].fill_between(grouped_data.index, grouped_data['lower'], grouped_data['upper'], color='blue', alpha=0.2, label='Variance')
    axs[2].set_xlabel('Discrete Time (days)')
    axs[2].set_ylabel('Profit per Second')
    axs[2].set_title('Average Profit per Second with Variance Over Time')
    axs[2].legend()

    for ax in axs[0:2]:
        ax.legend_.remove()


    handles, labels = axs[0].get_legend_handles_labels()


    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=len(handles))

    plt.tight_layout()
    plt.show()

def plot_params_data(data):
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))

    # if more than 10 trials, sample 10 equally spaced trials
    unique_trials = data['Trial'].unique()
    if len(unique_trials) > 10:
        sample_indices = np.linspace(0, len(unique_trials) - 1, 10, dtype=int)
        trials_to_plot = unique_trials[sample_indices]
        data = data[data['Trial'].isin(trials_to_plot)]

    params = ['Beta', 'Momentum', 'mBuy', 'c_a', 'c_r']
    for i, param in enumerate(params):
        sns.lineplot(ax=axs[i], data=data, x='Discrete Time (days)', y=param, hue='Trial')
        axs[i].set_xlabel('Discrete Time (days)')
        axs[i].set_ylabel(param)
        axs[i].set_title(f'{param} Over Time')

    plt.tight_layout()
    plt.show()



def plot_params_data_avg(data):
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    
    params = ['Beta', 'Momentum', 'mBuy', 'c_a', 'c_r']
    for i, param in enumerate(params):
        grouped_data = data.groupby('Discrete Time (days)')[param].agg(['mean', 'std'])
        grouped_data['upper'] = grouped_data['mean'] + grouped_data['std']
        grouped_data['lower'] = grouped_data['mean'] - grouped_data['std']
        
        axs[i].plot(grouped_data.index, grouped_data['mean'], label=f'Average {param}')
        axs[i].fill_between(grouped_data.index, grouped_data['lower'], grouped_data['upper'], color='blue', alpha=0.2, label='Variance')
        axs[i].set_xlabel('Discrete Time (days)')
        axs[i].set_ylabel(param)
        axs[i].set_title(f'Average {param} with Variance Over Time')
        axs[i].legend()

    plt.tight_layout()
    plt.show()


from scipy import stats
import pandas as pd

def extract_data_for_day(data, day):
    trial_dict = {}
    for trial in data['Trial'].unique():
        day_data = data[(data['Trial'] == trial) & (data['Discrete Time (days)'] == day)]
        profit = day_data['Profit per Second'].iloc[0] if not day_data.empty else None
        trial_dict[trial] = profit
    return trial_dict

def perform_normality_test(data):
    stat, p = stats.shapiro(data)
    return p > 0.05 

def perform_ttest_or_wilcoxon(data1, data2, alternative='greater'):

    if perform_normality_test(data1) and perform_normality_test(data2):
        t_stat, p_val = stats.ttest_rel(data1, data2, alternative=alternative)
        test_used = "t-test"
    else:
        t_stat, p_val = stats.wilcoxon(data1, data2, alternative=alternative)
        test_used = "Wilcoxon"

    null_hypothesis_finding = "Failed to reject H₀" if p_val > 0.05 else "Reject H₀"
    alternative_hypothesis_finding = "Not enough evidence" if p_val > 0.05 else "Evidence supports alternative"

    return t_stat, p_val, null_hypothesis_finding, alternative_hypothesis_finding, test_used

def extract_avg_data_for_range(data, day_range):
    trial_dict = {}
    for trial in data['Trial'].unique():
        range_data = data[(data['Trial'] == trial) & (data['Discrete Time (days)'] >= day_range[0]) & (data['Discrete Time (days)'] <= day_range[1])]
        avg_profit = range_data['Profit per Second'].mean() if not range_data.empty else None
        trial_dict[trial] = avg_profit
    return trial_dict

def perform_ttests_for_day_ranges(data, day_ranges, alternative: str ='greater'):
    ttest_results = []

    for i in range(len(day_ranges)):
        for j in range(i + 1, len(day_ranges)):
            range1_data = extract_avg_data_for_range(data, day_ranges[i])
            range2_data = extract_avg_data_for_range(data, day_ranges[j])

            range1_values = [avg_profit for avg_profit in range1_data.values() if avg_profit is not None]
            range2_values = [avg_profit for avg_profit in range2_data.values() if avg_profit is not None]

            if len(range1_values) > 1 and len(range2_values) > 1:
                t_stat, p_val, null_hypothesis_finding, alternative_hypothesis_finding, test_used = perform_ttest_or_wilcoxon(range2_values, range1_values, alternative=alternative)
                p_val_full = "{:.2e}".format(p_val)

                ttest_results.append({
                    'Comparison': f'Range of Days {day_ranges[j]} vs Range of Days{day_ranges[i]}',
                    'Test Used': test_used,
                    'Statistic': t_stat,
                    'p-value': p_val_full,
                    'Null Hypothesis': null_hypothesis_finding,
                    'Alternative Hypothesis': alternative_hypothesis_finding
                })

    return pd.DataFrame(ttest_results)

