import csv

# 1. Complete list of 868 PDBs
HEME_pdbs = [
    "1YZP", "1CRI", "4NVA", "4D3T", "3QM8", "2J18", "1VXA", "1PHA", "7RKR", "1DVE",
    "1DS4", "1SOG", "3P6N", "3WEC", "1D3S", "3FKG", "7PQ1", "3P6U", "8EWQ", "4NVN",
    "5HLQ", "1JIN", "1BEK", "6HQM", "2J19", "2CYM", "2ACP", "1MKR", "6GEQ", "1IRC",
    "5U5U", "6HQK", "8FDJ", "4NVG", "1GWU", "4ZDY", "3M8M", "1CCC", "1C53", "1H5H",
    "1YYD", "2BLI", "4G8U", "5CMV", "1BES", "4RM4", "1UX8", "6BD7", "5KD1", "5XXI",
    # Third batch added (150)
    "5KDY", "1JPB", "1BEM", "6LY4", "2MGA", "2EUQ", "2ASN", "1MLO", "6KRF", "1IYN", 
    "5VCE", "6LAA", "8ING", "4O6U", "1H57", "5B85", "3OZU", "1CCJ", "1CCB", "1H5Z", 
    "2AL0", "2CIW", "4ICT", "5EAG", "1BJE", "4UGF", "1VXF", "6E8Q", "5M6K", "6A7I", 
    "3O89", "2AQD", "3VOO", "5B50", "2MGG", "7NQN", "8EKO", "1A6G", "6RQ6", "7LS3", 
    "1NZ5", "5ZZG", "3OZZ", "2XVZ", "1NP4", "1Z8P", "9H1M", "6T0K", "2XKG", "1H5L", 
    "3E55", "1HRM", "3A4G", "7V46", "2Z3U", "5EJX", "2HZ2", "7N14", "1CP4", "6HQR", 
    "4AVD", "4NVF", "1LT0", "9Q4Q", "8S53", "3E2N", "1GEI", "4RLR", "2Q8Q", "7TEF", 
    "5HIW", "5G6B", "8GLZ", "8CCR", "3ABA", "4XV4", "1UBB", "6BDD", "1EHG", "1CPG", 
    "5OMU", "2BOQ", "6VZB", "2OHB", "1GEK", "8A8L", "8DJU", "1IWK", "3X34", "5IQX", 
    "7TSA", "3AWP", "1OFK", "3CCP", "2ZT4", "1YMA", "5UFG", "2IIZ", "6ATJ", "8VWK", 
    "5W58", "5M0N", "1EUP", "5ESL", "5IKG", "1PM1", "4NS2", "6HQQ", "1OXA", "4BLN", 
    "3E4N", "2JJO", "8SQP", "5KDZ", "5XKV", "4UG6", "2AMM", "5WK9", "2WHF", "7UF9", 
    "6U87", "6XAM", "1DP6", "2C7X", "7RL2", "7L3Y", "2VLY", "3IW1", "2J0P", "1ECD", 
    "1YRD", "8R9N", "8ZGO", "4UGR", "8F9H", "6EKZ", "2VKU", "1YWC", "5O1K", "4GRC", 
    "3HDL", "8GDI", "8R9Q", "5L1Q", "4APY", "1MNH", "2HZ3", "2EUU", "6MA8", "4NVL"
]

FAD_pdbs = [
    "8A1H", "1REO", "1DOB", "8Z45", "3NLC", "3FG2", "3AXB", "2AQJ", "8Z3G", "1PXC",
    "8JDY", "8Z44", "6RTM", "1OGI", "7OR2", "5GRT", "1E1N", "1E1L", "1OWP", "2YYM",
    "8CCL", "1TJ1", "1DOC", "9GN5", "3QFS", "3GYJ", "3DJJ", "2BAB", "9DTK", "1QNF",
    "8K41", "9F1Y", "6YRZ", "1OWN", "7RT0", "5KOW", "1E63", "1E39", "1PHH", "3COX",
    "3EF6", "6ICI", "7VJ0", "1DOE", "7C4N", "2XRY", "8Z26", "8JDG", "8X38", "5JCK",
    # Third batch added (150)
    "1B8S", "1GRE", "1N1P", "1TDE", "2AR8", "2I0K", "3DJG", "3GWD", "3QVR", "3T31", 
    "4H4S", "4U63", "5HXI", "5ZW8", "6SW2", "7ORZ", "7XPI", "8JDO", "8PLG", "8Z1J", 
    "8Z6I", "9KKG", "1BJK", "1GRG", "1NHS", "1TDF", "2CIE", "2IJG", "3DJL", "3GYI", 
    "3RP6", "3U5S", "4H4W", "4XXG", "5I39", "5ZYN", "6YS2", "7OSQ", "8B7S", "8JDV", 
    "8PXL", "8Z3D", "9C4P", "9ORN", "1BX1", "1GSN", "1NPX", "1U3D", "2CJC", "2JKC", 
    "3DK9", "3LVB", "3RP7", "3W2H", "4HA6", "4YKG", "5OGX", "6C7S", "7AV4", "7QU3", 
    "8BJY", "8JM1", "8QNB", "8Z3X", "9GXB", "9V6B", "1DOD", "1IQR", "1Q9I", "1V5E", 
    "2DJI", "2MBR", "3E2S", "3NG7", "3SX6", "3ZBU", "4K8D", "4YNU", "5UTH", "6DD6", 
    "7B02", "7V8R", "8C1U", "8JM2", "8QNC", "8Z41", "9HNK", "1E62", "1K0I", "1QGY", 
    "1V5G", "2DKH", "2R0C", "3FW1", "3NH3", "3SXI", "4D03", "4NZH", "4YWO", "5VW4", 
    "6KGQ", "7BR0", "7VJ5", "8C6B", "8JM4", "8UIQ", "8Z4K", "9HNL", "1FDR", "1K0L", 
    "1QUF", "1W35", "2GQW", "2YLS", "3G5S", "3NYE", "3SYI", "4D04", "4OVI", "5BUL", 
    "5WGY", "6LR8", "7KPQ", "7VJ6", "8CCM", "8JU4", "8WKC", "8Z4M", "9IQC", "1FNB", 
    "1M6I", "1R2J", "1WAM", "2GR2", "3D1C", "3GSI", "3O55", "3T0K", "4G1V", "4REK", 
    "5GV7", "5Y7A", "6PVI", "7KPT", "7VJB", "8ER1", "8PL6", "8YQM", "8Z4P", "9KKC"
]

ZN_2_pdbs = [
    "7G3B", "2H2I", "1FT7", "7Y2E", "4CXO", "3T87", "3PB9", "2WEG", "7UHI", "2E0P",
    "7HUX", "7XJ4", "6KM5", "1YSO", "6UGR", "5NS5", "1HFC", "1H3N", "2AW1", "3ORJ",
    "7G3R", "2H6S", "1FUA", "7Z0N", "4DEF", "3TGE", "3PN5", "2WHZ", "7VFR", "2E88",
    "7HVJ", "7YHD", "6LD4", "1Z1N", "6V4V", "5NYA", "1HKK", "1H71", "2B5W", "3OY0",
    "3RTT", "6CLD", "6YHE", "1G4J", "6PJV", "3KZZ", "7O2S", "7G68", "7JOB", "6LD1",
    # Third batch added (150)
    "1C2D", "1FUK", "1G52", "1H7P", "1HS6", "1HUG", "1IQU", "1KS7", "1LMH", "1OWP", 
    "1PE7", "1S0U", "1TBF", "1W9B", "1WKF", "1Z9Y", "2AXR", "2BHB", "2CBB", "2DYX", 
    "2EHT", "2EJC", "2HJN", "2PMP", "2RGJ", "2WHK", "2WXT", "2XEF", "3BKQ", "3CA2", 
    "3CZS", "3D09", "3EBG", "3FW1", "3ISI", "3L7X", "3LOV", "3N4B", "3NKM", "3OHR", 
    "3P58", "3P7P", "3P7S", "3PB6", "3Q1D", "3R4X", "3RG3", "3RZV", "3S1G", "3TT4", 
    "3TTC", "3U47", "4B9P", "4BJB", "4BM9", "4C4M", "4CA2", "4DZ9", "4EJM", "4EL4", 
    "4ELC", "4H2I", "4I0G", "4K7Z", "4MZI", "4N5P", "4P4E", "4R2J", "4RSY", "4U9B", 
    "4ZX4", "5AIJ", "5AMH", "5C3K", "5FLO", "5FNH", "5FNL", "5FNM", "5JC6", "5JMX", 
    "5JT5", "5NXV", "5OJH", "5ONY", "5PI7", "5PMX", "5QPD", "5SG4", "5SIM", "5SLD", 
    "5ZGW", "6BCC", "6CPA", "6DCH", "6I3E", "6IK4", "6JEB", "6LND", "6LUY", "6OIQ", 
    "6PDB", "6QSQ", "6QV5", "6RQU", "6SP0", "6TWA", "6U4T", "6VDN", "6W12", "6YO7", 
    "6YS2", "6YYZ", "7CI9", "7FU4", "7G3I", "7G4F", "7G4K", "7G59", "7G6W", "7H2Y", 
    "7H4E", "7HX1", "7I2I", "7I43", "7JO3", "7KUS", "7KZ7", "7MHH", "7OXD", "7PBA", 
    "7TSD", "7TW9", "7VJH", "7Y70", "8AA6", "8ATJ", "8C4W", "8CAL", "8CP7", "8OGF", 
    "8P95", "8PG5", "8PRC", "8QQA", "8Z4M", "9FTQ", "9G38", "9GCD", "9GU7", "9YC2"
]

CU_1_pdbs = [
    "2CAL", "1DZ0", "3QQX", "3F7L", "3DSO", "2FT7", "4MAI", "1OF0", "4DPC", "5SSZ",
    "5NQM", "3NT0", "1JZG", "4BTE", "2XV2", "1BXA", "1A8Z", "5MSZ", "2CAK", "7OG7",
    "5NQN", "2BZC", "1HAW", "2XMW", "2QDW", "2JCW", "2CCW", "1SF3", "4TM7", "1UUY",
    "5KBK", "4DP8", "1RJU", "6QVH", "5KBM", "2IDU", "5SSY", "5ARN", "6L9S", "5NQO",
    "3I9Z", "3F7K", "4DPA", "4F2F", "4P5S", "6R01", "4N3T", "2XV0", "5SSX", "1A3Z",
    # Third batch added (18)
    "1UUX", "2FK2", "2FT8", "2GBA", "2IDS", "2RAC", "3EIM", "3FSA", "3ZDW", "4DP1", 
    "4DP4", "4DP6", "4F2E", "4L05", "5C92", "5KBL", "6EK9", "6RW7"
]

# Cofactorless (200)
cofactorless_pdbs = [
    "2BQQ", "1FHG", "4TTP", "4G2E", "3SYJ", "2O6X", "1YU7", "1UIA", 
    "7IE4", "1HE9", "1GVL", "1WKA", "3R63", "3ZM2", "1FQI", "3JSN", 
    "7ID0", "3RLE", "7W8U", "4WJ1", "1BFG", "2WLW", "7IEG", "5R2I", 
    "4WEI", "2TMY", "3PTW", "5P7D", "1YRV", "1W7B", "6K1Y", "1XAA", 
    "5ZO0", "5RCS", "4OEF", "1KAB", "8C9L", "2FGT", "6J6E", "1SNQ", 
    "5DZ9", "6AO9", "3H2G", "1QG5", "1KWB", "3V75", "5C10", "1SZT", 
    "3ZPJ", "1YHW", "6K5M", "4WFU", "7Y5J", "6CB6", "2Y7N", "6F4M", 
    "5YDN", "3NPO", "4P9L", "1QTO", "3ASD", "4G14", "2YLH", "132L", 
    "6JUI", "4QRZ", "3RFY", "5P1Q", "1MJZ", "3VSR", "1HEP", "5NPT", 
    "6VV4", "4PMD", "1PEN", "3O7K", "5NJM", "3OUV", "9L1R", "6RU3",
    "5P18", "1MN4", "1CNU", "6HYN", "2NWD", "2F9F", "2CG7", "1QMR",
    "6G9P", "1LNS", "5R3B", "6HHN", "7X3H", "4U64", "1KF7", "5GLX",
    "3P2J", "1DST", "1DG3", "1L18", "2BMJ", "2DX1", "4KIA", "5I4W",
    "1CUC", "4YWF", "1YG2", "6AR0", "5P3O", "5Y30", "4U3V", "3OKQ",
    "2BTZ", "3UMF", "5G0Z", "2OBT", "7BAJ", "7Q6B", "1ANF", "6L2A",
    "7A8B", "1T7N", "5XCN", "3P7K", "3B5O", "2OBR", "1SLL", "2B2H",
    "8X8S", "6LYT", "3AA5", "1LMQ", "1L0B", "3GW3", "1L40", "3DQI",
    "7IHZ", "3BVS", "5IJM", "2ICC", "7AHW", "1EY0", "6EHX", "3ZDE",
    "4R1B", "1OEM", "9DD2", "8HEK", "3GS9", "1JAM", "4X1O", "2PW5",
    "7ICB", "5OYR", "5LZ1", "7VKP", "7NMO", "3E0E", "5DDV", "1XEI",
    "5YNL", "1I9Y", "1F1S", "5P6A", "2D27", "6P1F", "2PII", "9JLA",
    "1JI6", "7L6W", "2DYI", "7P25", "1LIB", "3GYW", "2OBS", "3VYC",
    "5P0D", "7IDX", "3EMI", "1TK1", "3FFM", "3DGJ", "1ZXQ", "5P8C",
    "2IQT", "5Y5D", "8P33", "5RDZ", "5P39", "1IIW", "5K2N", "5P07",
    "1UEK", "4QSC", "6EFR", "2F8H", "1TP0", "4ACO", "3GVQ", "2JDW"
]

# Combine into one master list
master_list = HEME_pdbs + FAD_pdbs + ZN_2_pdbs + CU_1_pdbs + cofactorless_pdbs
print(f"Total target PDBs: {len(master_list)}")

# 2. Extract PDBs from the first column of the CSV
csv_pdbs = set()
try:
    with open('Best_Features_Table.csv', mode='r') as f:
        reader = csv.reader(f)
        for row in reader:
            if row: # Skip empty rows
                # Extract the ID from column 1, clean whitespace/quotes, and make uppercase
                pdb_id = row[0].strip().replace('"', '').upper()
                csv_pdbs.add(pdb_id)
except FileNotFoundError:
    print("Error: Best_Features_Table.csv not found.")
    exit()

# 3. Check for missing codes using a for loop
missing_list = []
for pdb in master_list:
    if pdb.upper() not in csv_pdbs:
        missing_list.append(pdb)

# 4. Final Output
if not missing_list:
    print("Success! No PDB codes are missing.")
else:
    print(f"Found {len(missing_list)} missing PDBs:")
    for code in missing_list:
        print(code)

unique_master = set(pdb.upper() for pdb in master_list)
print(f"Unique target PDBs: {len(unique_master)}")

missing_list = [pdb for pdb in unique_master if pdb not in csv_pdbs]