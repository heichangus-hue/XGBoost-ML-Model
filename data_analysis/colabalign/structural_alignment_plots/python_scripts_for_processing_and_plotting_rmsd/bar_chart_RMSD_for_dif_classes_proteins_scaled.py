import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker


csv_file = 'batch_rmsd_summary_new.csv'
rmsd_data = pd.read_csv(csv_file)
print(rmsd_data.columns)

HEME = ["1yzp", "1cri", "4nva", "4d3t", "3qm8", "2j18", "1vxa", "1pha", "7rkr", "1dve", "1ds4", "1sog", "3p6n", "3wec", "1d3s", "3fkg", "7pq1", "3p6u", "8ewq", "4nvn"]
FAD = ["8a1h", "1reo", "1dob", "8z45", "3nlc", "3fg2", "3axb", "2aqj", "8z3g", "1pxc", "8jdy", "8z44", "6rtm", "1ogi", "7or2", "5grt", "1e1n", "1e1l", "1owp", "2yym"]
Zn_2 = ["7g3b", "2h2i", "1ft7", "7y2e", "4cxo", "3t87", "3pb9", "2weg", "7uhi", "2e0p", "7hux", "7xj4", "6km5", "1yso", "6ugr", "5ns5", "1hfc", "1h3n", "2aw1", "3orj"]
Cu_1 = ["2cal", "1dz0", "3qqx", "3f7l", "3dso", "2ft7", "4mai", "1of0", "4dpc", "5ssz", "5nqm", "3nt0", "1jzg", "4bte", "2xv2", "1bxa", "1a8z", "5msz", "2cak", "7og7"]
    
all_groups = [HEME, FAD, Zn_2, Cu_1]
group_names = ["HEME", "FAD", "Zn2+", "Cu+"]

for i in range(4):
    # Filter the data for the current group
    # .isin() checks if the PDB_Code in the CSV matches my list
    group_data = rmsd_data[rmsd_data['PDB_Code'].isin(all_groups[i])].copy()

    # Sort data from lowest to highest RMSD
    subset = group_data.sort_values('RMSD', ascending=False)

    # Plot via seaborn
    sns.barplot(x='PDB_Code', y='RMSD', data=subset, palette='inferno')
    plt.xlabel('PDB Code')
    plt.xticks(rotation=45)
    plt.ylabel('RMSD / $\mathrm{\AA}$')
    plt.ylim(0, 10)
    plt.title(f'Average RMSD for Proteins with {group_names[i]} Cofactor')
    plt.suptitle('Structural Average RMSD Comparison by Cofactor Class', size=14)
    plt.tight_layout()
    output_file = f'bar_chart_avg_{group_names[i]}_scaled.png'
    plt.savefig(output_file, dpi = 500)
    plt.close()







csv_file_2 = 'batch_max_rmsd_summary.csv'
rmsd_data_2 = pd.read_csv(csv_file_2)
print(rmsd_data_2.columns)

HEME = ["1yzp", "1cri", "4nva", "4d3t", "3qm8", "2j18", "1vxa", "1pha", "7rkr", "1dve", "1ds4", "1sog", "3p6n", "3wec", "1d3s", "3fkg", "7pq1", "3p6u", "8ewq", "4nvn"]
FAD = ["8a1h", "1reo", "1dob", "8z45", "3nlc", "3fg2", "3axb", "2aqj", "8z3g", "1pxc", "8jdy", "8z44", "6rtm", "1ogi", "7or2", "5grt", "1e1n", "1e1l", "1owp", "2yym"]
Zn_2 = ["7g3b", "2h2i", "1ft7", "7y2e", "4cxo", "3t87", "3pb9", "2weg", "7uhi", "2e0p", "7hux", "7xj4", "6km5", "1yso", "6ugr", "5ns5", "1hfc", "1h3n", "2aw1", "3orj"]
Cu_1 = ["2cal", "1dz0", "3qqx", "3f7l", "3dso", "2ft7", "4mai", "1of0", "4dpc", "5ssz", "5nqm", "3nt0", "1jzg", "4bte", "2xv2", "1bxa", "1a8z", "5msz", "2cak", "7og7"]
    
all_groups = [HEME, FAD, Zn_2, Cu_1]
group_names = ["HEME", "FAD", "Zn2+", "Cu+"]

for i in range(4):
    # Filter the data for the current group
    # .isin() checks if the PDB_Code in the CSV matches your list
    group_data = rmsd_data_2[rmsd_data_2['pdb_code'].isin(all_groups[i])].copy()

    # Sort data from lowest to highest RMSD
    subset = group_data.sort_values('max_RMSD', ascending=False)

    # Plot via seaborn
    sns.barplot(x='pdb_code', y='max_RMSD', data=subset, palette='inferno')
    plt.xlabel('PDB Code')
    plt.xticks(rotation=45)
    plt.ylabel('Maximum RMSD / $\mathrm{\AA}$')
    plt.ylim(0, 50)
    plt.title(f'Maximum RMSD for Proteins with {group_names[i]} Cofactor')
    plt.suptitle('Structural Maximum RMSD Comparison by Cofactor Class', size=14)
    plt.tight_layout()
    output_file = f'bar_chart_max_{group_names[i]}_scaled.png'
    plt.savefig(output_file, dpi = 500)
    plt.close()
